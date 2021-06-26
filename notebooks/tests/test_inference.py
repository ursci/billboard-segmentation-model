import sys
sys.path.append("../")
from inference import get_masked_pil_img
from PIL import Image

def test_get_masked_pil_img():
    output = get_masked_pil_img('../data/test/img/20201229Akihabara_out_00000_dir_020013selected_1.jpg')
    assert type(output) == Image
