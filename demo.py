import os
import argparse
import random

import gradio as gr
from nltk.tokenize import word_tokenize
from phonemizer import phonemize
from txtsplit import txtsplit

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert
from models import *
from text_utils import TextCleaner
from utils import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
textcleaner = TextCleaner()


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def preprocess(wave, mean=-4, std=4):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


class ModelPlaceholder:
    def __init__(self, config, device=None):
        # device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)  # model placeholder
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

    def inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = phonemize([f'… {text} …'], language="en-us", backend='espeak', with_stress=True, preserve_punctuation=True)
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = textcleaner(ps)
        if len(tokens) >= 510:
            raise gr.Error("Input is too Long")
        tokens.insert(0, 0)
        tokens.append(0)

        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s,  # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(
                d_en, s, input_lengths, text_mask
            )

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :]

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)


class ModelRepresentation:
    def __init__(self, root_dir, checkpoint_name):
        self._root_dir = root_dir
        self._checkpoint_name = checkpoint_name
        path_to_references = os.path.join(root_dir, "references")
        if os.path.exists(path_to_references):
            self._references = [x for x in os.listdir(os.path.join(root_dir, "references")) if x.endswith(".wav")]
        else:
            self._references = []

    @property
    def model_name(self) -> str:
        _, model_name = os.path.split(self._root_dir)
        return model_name

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self._root_dir, self._checkpoint_name)

    @property
    def references(self) -> list[str]:
        return self._references

    def __repr__(self):
        return f"ModelRepresentation(Name={self.model_name}, checkpoint_path={self.checkpoint_path})"


class ModelDirectory:
    def __init__(self, model_dir):
        """
        Walk through a directory look up for .pth file. If there is a .pth file, append a ModelRepresentation in a list.
        Return the list of ModelRepresentation.
        :param model_dir:
        :return:
        """
        models = {}
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".pth"):
                    model_repr = ModelRepresentation(root, file)
                    models.update({model_repr.model_name: model_repr})
        self.models = models

    @property
    def model_names(self):
        return [x for x in self.models]

    def __getitem__(self, model_name: str):
        return self.models[model_name]


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="Models")
    parser.add_argument("--libritts_config_path", type=str, default="Models/LibriTTS/config.yml")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == "__main__":
    args = cli_args()
    device = args.device
    config = yaml.safe_load(open(args.libritts_config_path))

    model = ModelPlaceholder(config)  # model placeholder
    ref_s = None  # style tensor placeholder

    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
    )

    model_dir = ModelDirectory(args.model_dir)

    with gr.Blocks(title="StyleTTS2", css="footer{display:none !important}", theme=theme) as demo:
        gr.Markdown("# Models")
        dropdown = gr.Dropdown(choices=model_dir.model_names, label="Select a Model")

        def update_model_params(model_name, progress_bar=gr.Progress()):
            progress_bar((0, None), desc="update_model_parameters")
            model_repr = model_dir[model_name]
            params_whole = torch.load(model_repr.checkpoint_path, map_location='cpu')
            params = params_whole['net']
            for step, key in enumerate(model.model):
                progress_bar((step, None), desc=f"updating x`{key}")
                if key in params:
                    try:
                        model.model[key].load_state_dict(params[key])
                    except:
                        from collections import OrderedDict
                        state_dict = params[key]
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v
                        model.model[key].load_state_dict(new_state_dict, strict=False)
                    model.model[key].eval()
            return gr.Dropdown(choices=model_repr.references + ["None"], label="Select a Audio"), gr.Dropdown(choices=model_repr.references + ["None"], label="Select a Audio")

        gr.Markdown("# Reference")
        with gr.Row():
            with gr.Column(scale=1):
                ref_1_dropdown = gr.Dropdown(value="None", choices=["None"], label="Select a Audio")
                ref_1 = gr.Audio(interactive=False, label="Reference One",
                                 waveform_options={'waveform_progress_color': '#3C82F6'})
            with gr.Column(scale=1):
                ref_2_dropdown = gr.Dropdown(value="None", choices=["None"], label="Select a Audio")
                ref_2 = gr.Audio(interactive=False, label="Reference Two",
                                 waveform_options={'waveform_progress_color': '#3C82F6'})

            def update_audio(model_name, audio_filename):
                model_repr = model_dir[model_name]
                return os.path.join(model_repr._root_dir, "references", audio_filename) if audio_filename != "None" else "None"

            ref_1_dropdown.input(update_audio, inputs=[dropdown, ref_1_dropdown], outputs=ref_1)
            ref_2_dropdown.input(update_audio, inputs=[dropdown, ref_2_dropdown], outputs=ref_2)

            with gr.Column(scale=1):
                operations = gr.Dropdown(choices=["add"], label="operations")

        dropdown.input(update_model_params, inputs=dropdown, outputs=[ref_1_dropdown, ref_2_dropdown])

        def synthesize(model_name, text, ref_1, ref_2, operations, lngsteps, progress=gr.Progress()):
            model_repr = model_dir[model_name]
            if ref_1 == "None":
                raise gr.Error("Please select a Reference")
            else:
                style_1 = model.compute_style(os.path.join(model_repr._root_dir, "references", ref_1))
            if ref_2 != "None":
                style_2 = model.compute_style(os.path.join(model_repr._root_dir, "references", ref_2))
                if operations == "add":
                    ref_s = (style_1 + style_2) / 2
                else:
                    raise gr.Error("Please select a operations")
            else:
                ref_s = style_1

            if text.strip() == "":
                raise gr.Error("You must enter some text")
            if len(text) > 50000:
                raise gr.Error("Text must be <50k characters")

            texts = txtsplit(text)
            audios = []
            for text in progress.tqdm(texts):
                audios.append(
                    model.inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1)
                )
            return 24000, np.concatenate(audios)

        gr.Markdown("# Generation")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(label="Text", info="Suggest input a full sentence.", interactive=True)
                diffusion_steps = gr.Slider(minimum=3, maximum=50, value=30, step=1, label="Diffusion Steps",
                                              info="The more the better but the slower it is", interactive=True)
            with gr.Column(scale=1):
                audio = gr.Audio(interactive=False, label="Generated Audio",
                                 waveform_options={'waveform_progress_color': '#3C82F6'})
                btn = gr.Button("Generate", variant="primary")
                btn.click(synthesize, inputs=[dropdown, inp, ref_1_dropdown, ref_2_dropdown, operations, diffusion_steps], outputs=[audio])

    demo.queue(api_open=False, max_size=15).launch(show_api=False)
