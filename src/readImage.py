#from PIL import Image
#import requests
#from io import BytesIO

#url = "https://upload.wikimedia.org/wikipedia/commons/9/92/Brookings.jpg"

#response = requests.get(url)
#img = Image.open(BytesIO(response.content))

#print(img)

#picLocation = 'C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\train\\400X400\\'+image_id+'.jpg'

def showImage(loc):
    import matplotlib as plt
    from matplotlib import pyplot 
    im = pyplot.imread(loc)
    pyplot.imshow(im)

# showImage(picLocation)
