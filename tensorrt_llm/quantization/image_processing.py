# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


class BaseImageProcessor:

    def __init__(self, tokenizer, device='auto'):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, **kwargs):
        return self.tokenizer(**kwargs)

    def preprocess_function(self, examples):
        raise NotImplementedError(
            "Each image processor must implement its own preprocess method")

    def collate_function(self, examples):
        raise NotImplementedError(
            "Each image processor must implement its own colloate method")


# A light Encapsulation for Huggingface MllamaImageProcessor
class MllamaImageProcessor(BaseImageProcessor):

    def preprocess_function(self, examples):
        # Prepare prompts in a generic chat format
        if 'question' in examples:
            question = examples['question']
        else:
            question = "Describe this image."

        if examples['image'] is not None:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role":
                        "user",
                        "content": [{
                            "type": "image"
                        }, {
                            "type": "text",
                            "text": question
                        }],
                    }],
                    add_generation_prompt=True,
                )
            else:
                prompt = f"<|image|><|begin_of_text|>{question}"

            # Process images using the processor's image processor
            values = self.tokenizer(text=prompt,
                                    images=examples['image'],
                                    return_tensors="pt").to(self.device)
        else:
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": question
                        }],
                    }],
                    add_generation_prompt=True,
                )
            else:
                prompt = question

            values = self.tokenizer(text=prompt,
                                    images=None,
                                    return_tensors="pt").to(self.device)

            values['pixel_values'] = None
            values['aspect_ratio_ids'] = None
            values['aspect_ratio_mask'] = None
            values['cross_attention_mask'] = None

        return values

    # Define a collate function to process images during data loading
    def collate_function(self, batch):
        batch[0]['input_ids'] = torch.LongTensor(batch[0]['input_ids']).to(
            self.device)
        batch[0]['attention_mask'] = torch.LongTensor(
            batch[0]['attention_mask']).to(self.device)

        if batch[0]['pixel_values'] is not None:
            batch[0]['pixel_values'] = torch.Tensor(
                batch[0]['pixel_values']).to(self.device)
            batch[0]['aspect_ratio_ids'] = torch.LongTensor(
                batch[0]['aspect_ratio_ids']).to(self.device)
            batch[0]['aspect_ratio_mask'] = torch.LongTensor(
                batch[0]['aspect_ratio_mask']).to(self.device)
            batch[0]['cross_attention_mask'] = torch.LongTensor(
                batch[0]['cross_attention_mask']).to(self.device)

        return batch[0]


# InternVL preprocessing uses ImageNet mean/std (as its InternViT tower expects).
INTERNVL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
INTERNVL_IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=INTERNVL_IMAGENET_MEAN, std=INTERNVL_IMAGENET_STD),
    ])


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width,
                                        height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image,
                                 min_num=1,
                                 max_num=12,
                                 image_size=448,
                                 use_thumbnail=False):
    # Standard InternVL tiling: pick the tile grid whose aspect ratio best
    # matches the image, crop into image_size x image_size tiles, optionally
    # append a downscaled thumbnail of the whole image.
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j)
         for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    cols = target_width // image_size
    processed_images = []
    for i in range(blocks):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def internvl_load_image(image,
                        input_size=448,
                        min_num=1,
                        max_num=12,
                        use_thumbnail=True):
    """Return InternVL pixel_values of shape [num_tiles, 3, input_size, input_size]."""
    transform = _internvl_build_transform(input_size)
    tiles = _internvl_dynamic_preprocess(image,
                                         min_num=min_num,
                                         max_num=max_num,
                                         image_size=input_size,
                                         use_thumbnail=use_thumbnail)
    return torch.stack([transform(tile) for tile in tiles])


# A self-contained InternVL calibration processor. InternVL ships no combined HF
# AutoProcessor (and often no AutoImageProcessor), so this reproduces InternVL's
# own dynamic-tiling transform and expands the <IMG_CONTEXT> placeholder itself.
class InternVLImageProcessor(BaseImageProcessor):
    # Special tokens marking where image features are spliced into the sequence.
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    def __init__(self,
                 tokenizer,
                 num_image_token,
                 image_size=448,
                 min_num=1,
                 max_num=12,
                 use_thumbnail=True,
                 messages=None,
                 device='auto',
                 dtype=torch.bfloat16):
        super().__init__(tokenizer, device)
        # Number of vision embeddings produced per image tile; the language
        # model expects exactly this many <IMG_CONTEXT> placeholders per tile.
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.min_num = min_num
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail
        self.dtype = dtype
        # InternVL's language model has no fallback for missing images, so
        # calibration must keep every sample multimodal. get_calib_dataloader
        # honors this flag to drop text-only rows.
        self.requires_image = True
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(
            self.IMG_CONTEXT_TOKEN)
        # Pre-render a chat template holding a single "<image>" placeholder that
        # preprocess_function expands per-sample. None -> a plain image+question
        # prompt is used instead (e.g. for ScienceQA).
        self.prompt_template = self._build_prompt_template(messages)

    @staticmethod
    def _build_prompt_template(messages):
        if not messages:
            return None
        parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if isinstance(content, list):
                text = ''
                for item in content:
                    itype = item.get('type', '')
                    if 'image' in itype:
                        text += '<image>'
                    elif itype == 'text':
                        text += item.get('text', '')
                content = text
            parts.append((role, content))
        rendered = ''
        for idx, (role, content) in enumerate(parts):
            if idx == len(parts) - 1 and role == 'assistant':
                # Trailing assistant turn is a generation prefix (no <|im_end|>).
                rendered += f'<|im_start|>{role}\n{content}'
            else:
                rendered += f'<|im_start|>{role}\n{content}<|im_end|>\n'
        if parts[-1][0] != 'assistant':
            rendered += '<|im_start|>assistant\n'
        return rendered

    def __call__(self, **kwargs):
        return self.tokenizer(**kwargs)

    def _load_image(self, examples):
        from PIL import Image
        if examples.get('image_path'):
            return Image.open(examples['image_path']).convert('RGB')
        image = examples.get('image', None)
        if image is not None:
            return image.convert('RGB') if hasattr(image, 'convert') else image
        # Defensive fallback; multimodal calibration should never hit this.
        return Image.new('RGB', (self.image_size, self.image_size))

    def preprocess_function(self, examples):
        image = self._load_image(examples)
        pixel_values = internvl_load_image(image,
                                           input_size=self.image_size,
                                           min_num=self.min_num,
                                           max_num=self.max_num,
                                           use_thumbnail=self.use_thumbnail)
        num_patches = pixel_values.shape[0]

        # Expand into (num_image_token * num_patches) context tokens so the count
        # matches the flattened vision embeddings the language model splices in.
        image_block = (
            self.IMG_START_TOKEN +
            self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches +
            self.IMG_END_TOKEN)
        # Prompt precedence: a per-sample "prompt" column (a rendered chat
        # template holding a single "<image>" placeholder), then a fixed template
        # supplied at construction, then a plain image+question fallback.
        template = examples.get('prompt') or self.prompt_template
        if template is not None:
            prompt = template.replace('<image>', image_block)
        else:
            question = examples['question'] if examples.get(
                'question') else 'Describe this image.'
            prompt = (f'<|im_start|>user\n{image_block}\n{question}<|im_end|>\n'
                      f'<|im_start|>assistant\n')

        # The prompt already carries the chat special tokens, so do not let the
        # tokenizer add its own.
        values = self.tokenizer(prompt,
                                return_tensors="pt",
                                add_special_tokens=False)
        values['pixel_values'] = pixel_values
        values['image_flags'] = torch.ones((num_patches, 1), dtype=torch.long)
        return values

    def collate_function(self, batch):
        batch[0]['input_ids'] = torch.LongTensor(batch[0]['input_ids']).to(
            self.device)
        batch[0]['attention_mask'] = torch.LongTensor(
            batch[0]['attention_mask']).to(self.device)
        batch[0]['pixel_values'] = torch.tensor(batch[0]['pixel_values'],
                                                dtype=self.dtype).to(
                                                    self.device)
        batch[0]['image_flags'] = torch.LongTensor(batch[0]['image_flags']).to(
            self.device)

        return batch[0]
