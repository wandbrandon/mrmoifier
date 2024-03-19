import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity
from PIL import Image
import PIL.ImageOps as image_ops


def generate_rotated_tiles(tile):
    """
    Generate all rotated forms (0, 90, 180, 270 degrees) of a given tile.
    """
    rotations = [0, 90, 180, 270]
    rotated_tiles = [tile.rotate(angle) for angle in rotations]
    return rotated_tiles

def load_and_slice_tileset(tileset_path):
    """
    Load the tileset image, slice it into 8x8 tiles, and generate all rotated forms for each tile.
    """
    tileset_img = Image.open(tileset_path).convert('RGB')
    total_width, total_height = tileset_img.size
    print("Width and Height of Tileset:", total_width, total_height)
    tiles = []

    for i in range(0, total_height, 8):
        for j in range(0, total_width, 8):
            box = (j, i, j+8, i+8)
            tile = tileset_img.crop(box)
            rotated_tiles = generate_rotated_tiles(tile)
            inverted_tiles = [image_ops.invert(tile) for tile in rotated_tiles]
            tiles.extend([*rotated_tiles, *inverted_tiles])

    print("Length of Tiles: ", len(tiles))
    return [np.array(tile) for tile in tiles]

def compare_tile_to_tileset(input_tile: Image, tileset):
    """
    Compare the input tile to each tile in the tileset (including rotated forms) using SSIM.
    """

    input_tile_color = input_tile.convert('P', palette=Image.ADAPTIVE, colors=2).convert('RGB')
    input_tile_color_arr = np.array(input_tile_color)

    input_tile_arr = np.array(input_tile)

    max_ssim_score = -1
    max_ssim_score_tile = None
    for tile in tileset:
        colored_tile = compute_colored_tile(tile, input_tile_color_arr)
        # Image.fromarray(tile).show(title="Current Tile")
        # Image.fromarray(input_tile_arr).show(title="Original Tile")
        # Image.fromarray(input_tile_color_arr).show(title="Reduced Color Tile")
        # Image.fromarray(colored_tile).show("New Title")
        score = structural_similarity(input_tile_arr, colored_tile, multichannel=True, win_size=7, channel_axis=2)
        if max_ssim_score < score:
            max_ssim_score = score
            max_ssim_score_tile = colored_tile
        if max_ssim_score == 1:
            break
    return max_ssim_score_tile, max_ssim_score

def compute_colored_tile(tile, input_tile_arr):
    """
    Compute the colored tile by using the input tile's color.
    """

    #get the indicies of all black pixels in the tile
    black_pixels = np.where(np.all(tile == [0, 0, 0], axis=-1))

    #get the indicies of all white pixels in the tile
    white_pixels = np.where(np.all(tile == [255, 255, 255], axis=-1))

    #get the average color of the selected white pixels in the input tile
    average_color_white = stats.mode(input_tile_arr[white_pixels], axis=0).mode

    #get the average color of the selected black pixels in the input tile
    average_color_black = stats.mode(input_tile_arr[black_pixels], axis=0).mode

    color_tile = tile.copy()

    color_tile[black_pixels] = average_color_black
    color_tile[white_pixels] = average_color_white

    return color_tile

def process_image(image_path, tileset_path):
    """
    Process the input image by comparing each 8x8 segment to each tile in the tileset (and its rotations).
    """
    image = Image.open(image_path).convert('RGB')
    tileset = load_and_slice_tileset(tileset_path)
    new_image = Image.new('RGB', image.size, (0, 0, 0))
    
    width, height = image.size
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            box = (j, i, j+8, i+8)
            input_tile = image.crop(box)
            replace_tile, ssim_score = compare_tile_to_tileset(input_tile, tileset)
            # Here, you can process SSIM scores, for example, to find the highest score
            print(f"Tile at ({j}, {i}) Max SSIM scores: {ssim_score}")
            new_image.paste(Image.fromarray(replace_tile), box)

    new_image.show()
    return new_image


# Example usage
process_image("/Users/brandonwand/Documents/projects/dreambound/python-libraries/mrmo_converter/assets/test.png", "/Users/brandonwand/Documents/projects/dreambound/python-libraries/mrmo_converter/assets/mrmotext-reduced.png")
