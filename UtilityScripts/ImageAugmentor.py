import os
from PIL import Image
from torchvision import transforms

def augment_images(directory, n):
    """
    Augments each image in the specified directory n times using random transformations.

    Parameters:
    - directory (str): The path to the directory containing the images.
    - n (int): The number of times to augment each image.

    The augmented images are saved in the same directory with unique filenames.
    """

    # Define the transformation pipeline
    def transform_image(image_size):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((15, 180)),
            transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
            # transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
            transforms.ColorJitter(brightness=(0.5, 1.2)),
            transforms.ToTensor(),  # Convert image to tensor
            transforms.ToPILImage()  # Convert back to PIL image
            ])
        return transform

    # Process each image in the directory
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            filepath = os.path.join(directory, filename)
            image = Image.open(filepath)

            # Apply the augmentation n times
            for i in range(n):
                width, height = image.size
                transform = transform_image((height, width))
                augmented_image = transform(image)
                augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                augmented_image.save(os.path.join(directory, augmented_filename))

augment_images("C:/Users/tjmaz/OneDrive/Desktop/GitHub/CoinCounter/UnannotatedImages", 5)
