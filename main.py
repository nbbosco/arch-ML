from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite


model = tflite.Interpreter("static/model_arch_10.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

class_mapping = {0: 'Achaemenid architecture',
 1: 'American Foursquare architecture',
 2: 'American craftsman style',
 3: 'Ancient Egyptian architecture',
 4: 'Art Deco architecture',
 5: 'Art Nouveau architecture',
 6: 'Baroque architecture',
 7: 'Bauhaus architecture',
 8: 'Beaux-Arts architecture',
 9: 'Byzantine architecture',
 10: 'Chicago school architecture',
 11: 'Colonial architecture',
 12: 'Deconstructivism',
 13: 'Edwardian architecture',
 14: 'Georgian architecture',
 15: 'Gothic architecture',
 16: 'Greek Revival architecture',
 17: 'International style',
 18: 'Novelty architecture',
 19: 'Palladian architecture',
 20: 'Postmodern architecture',
 21: 'Queen Anne architecture',
 22: 'Romanesque architecture',
 23: 'Russian Revival architecture',
 24: 'Tudor Revival architecture'}

def model_predict(images_arr):
    predictions = [0] * len(images_arr)

    for i, val in enumerate(predictions):
        model.set_tensor(input_details[0]['index'], images_arr[i].reshape((1, 150, 150, 3)))
        model.invoke()
        predictions[i] = model.get_tensor(output_details[0]['index']).reshape((25,))
    
    prediction_probabilities = np.array(predictions)
    argmaxs = np.argmax(prediction_probabilities, axis=1)

    return argmaxs


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def resize(image):
  return cv2.resize(image, (150, 150))

@app.post("/uploadfiles/", response_class=HTMLResponse)
async def create_upload_files(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        f = await file.read()
        images.append(f)
    
    
    images = [np.frombuffer(img, np.uint8) for img in images]
    images = [cv2.imdecode(img, cv2.IMREAD_COLOR) for img in images]
    images_resized = [resize(img) for img in images]
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_resized]

    names = [file.filename for file in files]

    for image, name in zip(images_rgb, names):
        pillow_image = Image.fromarray(image)
        pillow_image.save('static/' + name)
    
    image_paths = ['static/' + name for name in names]

    images_arr = np.array(images_rgb, dtype=np.float32)

    class_indexes = model_predict(images_arr)

    class_predictions = [class_mapping[x] for x in class_indexes]

    column_labels = ["Building", "Prediction"]

    table_html = get_html_table(image_paths, class_predictions, column_labels)

    content = head_html + """
    <marquee width="525" behavior="alternate"><h1 style="color:#7C99AC;font-family:Roboto">Style Prediction</h1></marquee>
    """ + str(table_html) + '''<br><form method="post" action="/">
    <button type="submit">Home</button>
    </form>'''

    return content


@app.post("/", response_class=HTMLResponse)
@app.get('/', response_class=HTMLResponse)
async def main():
    content = head_html + """
    <marquee width="525" behavior=alternate"><h1 style="color:#7C99AC; font-family:Roboto">Architecture Style - Machine Learning</h1></marquee>
    <h3 style="color:#92A9BD; font-family:Roboto">Upload an image of the building to predict his style</h3><br>
    """
    original_paths = ['0.jpg', '1.jpg', '2.jpg', 
                      '3.jpg', '4.jpg', '5.JPG',
                      '6.JPG', '7.JPG', '8.jpg',
                      '9.jpg', '10.JPG', '11.JPG',
                      '12.jpg', '13.jpg', '14.JPG',
                      '15.jpg', '16.jpg', '17.jpg',
                      '18.jpg', '19.JPG', '20.jpg',
                      '21.jpg', '22.jpg', '23.JPG',
                      '24.jpg']

    full_original_paths = ['static/original/' + x for x in original_paths]

    display_names = ['Achaemenid architecture','American Foursquare architecture','American craftsman style','Ancient Egyptian architecture',
    'Art Deco architecture','Art Nouveau architecture','Baroque architecture','Bauhaus architecture','Beaux-Arts architecture','Byzantine architecture',
    'Chicago school architecture','Colonial architecture','Deconstructivism','Edwardian architecture','Georgian architecture','Gothic architecture',
    'Greek Revival architecture','International style','Novelty architecture','Palladian architecture','Postmodern architecture','Queen Anne architecture',
    'Romanesque architecture','Russian Revival architecture','Tudor Revival architecture']

    column_labels = []
    
    contentInput = """
    <br/>
    <form  action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """

    contentFooter = """
    <br/>
    <br/>
    <h5 style="color:#92A9BD; font-family:Roboto">@Creator: nbbosco - Follow my projects: </h5><a> https://github.com/nbbosco </a><br>
    </body>
    """

    content = content + contentInput + get_html_table(full_original_paths, display_names, column_labels) + contentFooter

    return content

head_html = """
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background-color:#D3DEDC;">
<center>
"""

def get_html_table(image_paths, names, column_labels):
    s = '<table align="center">'
    if column_labels:
        s += '<tr><th><h4 color:#7C99AC; style="font-family:Roboto">' + column_labels[0] + '</h4></th><th><h4 color:#92A9BD; style="font-family:Roboto">' + column_labels[1] + '</h4></th></tr>'
    
    for name, image_path in zip(names, image_paths):
        s += '<tr><td><img height="80" width="100" src="/' + image_path + '" ></td>'
        s += '<td style="text-align:center">' + name + '</td></tr>'
    s += '</table>'
    
    return s