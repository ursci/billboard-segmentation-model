import json, glob, os, sys
from PIL import Image, ImageDraw


def generate_mask_image(json_file_path, output_dir):
	fp = open(json_file_path, "r")
	json_data = json.load(fp)
	fp.close()

	image_height = json_data["size"]["height"]
	image_width = json_data["size"]["width"]
	mask_image = Image.new("RGBA", (image_width, image_height), 'black')
	draw = ImageDraw.Draw(mask_image)

	for obj in json_data["objects"]:
		if obj["geometryType"] == "polygon":
			draw.polygon([(point[0],point[1]) for point in obj["points"]["exterior"]], fill = "white", outline = "white")
		else:
			print(obj["geometryType"])

	mask_image.save(output_dir + "/" + json_file_path.split("/")[-1].split("\\")[-1] + ".png")


def generate_all_annotation_mask_image(input_dir, output_dir):
	for json_file_path in glob.glob(input_dir + "/*.json"):
		print(json_file_path)
		generate_mask_image(json_file_path, output_dir)


if __name__ == '__main__':
	try:
		input_dir = sys.argv[1]
		output_dir = sys.argv[2]
	except IndexError:
		print("Not enough arguments.\r\nUse this code like below.\r\n\r\npython mask_generator.py <input_directory> <output_directory>\r\n")
		sys.exit(-1)

	os.makedirs(output_dir, exist_ok = True)
	generate_all_annotation_mask_image(input_dir, output_dir)
	