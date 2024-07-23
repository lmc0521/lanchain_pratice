from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

image = pipeline("a man is driving a car").images[0]
image.save("astronaut_rides_horse.png")
image.show()