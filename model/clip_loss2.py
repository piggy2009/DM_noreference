import torch
from torch import nn
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel
# from model.spatial_attention import SpatialTransformer
import clip
import random



def choose_clip_prompts(main_object):
    options = ['natural ' + main_object + ' on land',
               'the ' + main_object + ' is in an underwater condition, make it clearer and show it in the air',
               'remove the underwater color noise of the ' + main_object + ' in the underwater']
    selected = random.choice(options)
    return selected

def generate_src_target_txt(labels):
    src_txt_list = []
    target_txt_list = []
    for i in range(labels.shape[0]):
        label = labels[i].data.cpu().numpy()
        if label == 0:
            src_txt = 'fishes in a noisy the underwater scene'
            # target_txt = 'beautiful fishes in a clean underwater scene'
            target_txt = choose_clip_prompts('fishes')
        elif label == 1:
            src_txt = 'marine life in a noisy the underwater scene'
            # target_txt = 'beautiful marine life in a clean underwater scene'
            target_txt = choose_clip_prompts('marine life')
        elif label == 2:
            src_txt = 'coral in the underwater'
            # target_txt = 'beautiful coral in a clean underwater scene'
            target_txt = choose_clip_prompts('coral')
        elif label == 3:
            src_txt = 'rock in the underwater'
            # target_txt = 'rock in a clean underwater scene'
            target_txt = choose_clip_prompts('rock')
        elif label == 4:
            src_txt = 'diving people in the underwater'
            # target_txt = 'diving people in a clean underwater scene'
            target_txt = choose_clip_prompts('diving people')
        elif label == 5:
            src_txt = 'deep see scenes'
            # target_txt = 'clear underwater scenes'
            target_txt = choose_clip_prompts('object')
        elif label == 6:
            src_txt = 'wreckage in the underwater'
            # target_txt = 'wreckage in a clean underwater scene'
            target_txt = choose_clip_prompts('wreckage')
        elif label == 7:
            src_txt = 'sculpture in the underwater'
            # target_txt = 'sculpture in a clean underwater scene'
            target_txt = choose_clip_prompts('sculpture')
        elif label == 8:
            src_txt = 'caves in the underwater'
            # target_txt = 'normal caves in a clean underwater scene'
            target_txt = choose_clip_prompts('caves')
        else:
            src_txt = 'underwater stuff'
            # target_txt = 'underwater stuff in a clean underwater scene'
            target_txt = choose_clip_prompts('stuff')
        src_txt_list.append(src_txt)
        target_txt_list.append(target_txt)
        # target_txt_list.append('')
    return src_txt_list, target_txt_list

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


if __name__ == '__main__':
    # clip_loss = CLIP_loss()
    # path = '/home/ty/code/DM_underwater/dataset/water_train_16_128/hr_128_real/00001.png'
    # from torchvision import transforms
    # # image = Image.open(path).convert("RGB")
    # # image = transforms.ToTensor()(image).unsqueeze(0).to('cuda:0')
    # image = torch.randn([2, 3, 128, 128]).to('cuda:0')
    # image.clip_(-1, 1)
    # labels = torch.tensor([[1], [2]]).to('cuda:0')
    # clip_loss(image, image, labels)
    clip = FrozenCLIPEmbedder().to('cuda:0')
    a = ['underwater', 'air']
    output = clip.encode(a)

    print(output.shape)
    # trans = SpatialTransformer(320, 1, 320, context_dim=768).to('cuda:0')
    # input = torch.randn([1, 320, 64, 64]).to('cuda:0')
    # a = trans(input, output)
    # print(a.shape)




