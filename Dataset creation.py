#Toy dataset creation. 

#Libraries
from PIL import Image, ImageDraw
from random import *
from math import sin, cos, radians, pi
import numpy as np
from skimage.util import random_noise
import os, sys, shutil
import pandas as pd
import splitfolders

#Parameters
#For 128x128, set width to 5 and Extra_pix to 1, Radius to 6
S = 128 #Size of the image
WIDTH = 5 #Line width in pixels for each shape
EXTRA_PIX = 1 #Number of pixels to add for the test image to each side
RADIUS = 6
NOISE = 0.75 #Change to 0, 0.25, 0.5, 0.75 and 1.0

NOISE_DIR = int(NOISE*100)
seed(50) #Seed for reproducibility

IMG_SIZE = (S,S)
BACKGROUND = 'white' #Background color for the images
N_EXAMPLES = 1000 #Number of examples for EACH class. N_EXAMPLES*4 is the total number of examples


#Csv and folder
parent_dir = 'D:/Documentos/Facultad/Bionics M. Sc/03. Final Semester/Master Thesis/Main Program/Data'
data_folder = str(NOISE_DIR) + '_noise/'
test_folder = str(NOISE_DIR) + '_noise_test/'
output_folder = str(NOISE_DIR) + '_noise_data/'
try:
    os.mkdir(data_folder)
except FileExistsError:
    # directory already exists
    pass
try:
    os.mkdir(test_folder)
except FileExistsError:
    # directory already exists
    pass

square_dir = data_folder + "square/"
circle_dir = data_folder + "circle/"
triangle_dir = data_folder + "triangle/"
trapezoid_dir = data_folder + "trapezoid/"
empty_dir = data_folder + "empty/"
try:
    os.mkdir(square_dir)
except FileExistsError:
    # directory already exists
    pass
try:
    os.mkdir(circle_dir)
except FileExistsError:
    # directory already exists
    pass
try:
    os.mkdir(triangle_dir)
except FileExistsError:
    # directory already exists
    pass
try:
    os.mkdir(trapezoid_dir)
except FileExistsError:
    # directory already exists
    pass
try:
    os.mkdir(empty_dir)
except FileExistsError:
    # directory already exists
    pass

csv_file = str(NOISE_DIR) + '_noise.csv'
data = {"file_dir":[],"file_name":[],"class":[],"coordinates":[], "noise":[]}

#Coordinates Functions
def randTriangle():
    
  x1, y1 = randint(int(-0.1*S),int(0.6*S)), randint(int(0.2*S),int(0.6*S))
  angle =radians(-60)
  rotation = radians(randint(0,90))
  side = randint(int(0.2*S),int(0.6*S))
  x2 = x1 + cos(rotation)*side
  y2 = y1 + sin(rotation)*side

  x3 = x1 + cos(angle+rotation)*side
  y3 = y1 + sin(angle+rotation)*side
  points = [(x1,y1), (x2,y2), (x3,y3), (x1, y1)]
  coordinates = [(x1,y1), (x2,y2), (x3,y3)]
  return points, coordinates

def randRect():
  x1, y1 = randint(int(-0.05*S),int(0.6*S)), randint(int(0.4*S),int(0.6*S))

  side = randint(int(0.2*S),int(0.55*S))

  rotation = radians(randint(0,90))

  x2 = x1 + cos(rotation)*side
  y2 = y1 + sin(rotation)*side

  x3 = x2 + sin(rotation)*side
  y3 = y2 - cos(rotation)*side

  x4 = x3 - cos(rotation)*side
  y4 = y3 - sin(rotation)*side
  points = [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x1,y1)]
  coordinates = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
  return points, coordinates

def randTrap():
  x1, y1 = randint(int(-0.05*S),int(0.6*S)), randint(int(0.4*S),int(0.6*S))

  side_up = randint(int(0.2*S),int(0.55*S))
  side_down = randint(int(0.2*S),int(0.55*S))
  while (np.abs(side_up-side_down)<=10):
    side_up = randint(int(0.2*S),int(0.55*S))
    side_down = randint(int(0.2*S),int(0.55*S))
  height = randint(int(0.2*S),int(0.55*S))
  rotation = radians(randint(0,90))
  angle = radians(randint(45,75))

  x2 = x1 + cos(rotation)*side_up
  y2 = y1 + sin(rotation)*side_up

  x4 = x1 + cos(rotation-angle)*height
  y4 = y1 + sin(rotation-angle)*height

  x3 = x4 + cos(rotation)*side_down
  y3 = y4 + sin(rotation)*side_down
  points = [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x1,y1)]
  coordinates = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
  return points, coordinates

def randCirc():
  x1 = randint(0,S)
  y1 = randint(0,S)
  side = randint(int(0.2*S),int(0.7*S))

  if (x1+side)<S:
    x2 = x1 + side
  else:
    x2 = x1
    x1 = x2 - side

  if (y1+side)<S:
    y2 = y1 + side
  else:
    y2 = y1
    y1 = y2 - side

  points = [(x1, y1), (x2, y2)]
  coordinates = (x1, y1, x2, y2, side)

  return points, coordinates

#Noise function
def add_noise(image):
  im_arr = np.asarray(image)
  noise_img = random_noise(im_arr, mode='s&p', amount = NOISE)
  noise_img = (255*noise_img).astype(np.uint8)
  img = Image.fromarray(noise_img)
  noise_type = "salt & pepper"
  return img, noise_type

#Color function
def get_color():
  R = 255
  G = 255
  B = 255
  while(R+G+B>=690):
    R = randint(0,255)
    G = randint(0,255)
    B = randint(0,255)

  fill = (R,G,B,randint(128,255))
  return fill

#Shapes functions
def square(output_path, edge_path, vertex_path):
    #Square creation
    image = Image.new("RGB", IMG_SIZE, BACKGROUND)
    points, coordinates = randRect()
    draw = ImageDraw.Draw(image)
    draw.line(points, fill = get_color(), width = WIDTH)
    image, noise_type = add_noise(image)
    data["noise"].append(noise_type)
    data["coordinates"].append(coordinates)
    image.save(output_path)

    #Edge Test
    edge_test = Image.new("L", IMG_SIZE, 'black')
    draw_edge = ImageDraw.Draw(edge_test)
    draw_edge.line(points, fill = (255), width = (WIDTH + 2*EXTRA_PIX))
    edge_test.save(edge_path)

    #Vertex Test
    vertex_test = Image.new("L", IMG_SIZE, 'black')
    draw_vertex = ImageDraw.Draw(vertex_test)
    (x1, y1) = points[0]
    (x2, y2) = points[1]
    (x3, y3) = points[2]
    (x4, y4) = points[3]
    
    draw_vertex.ellipse((x1-RADIUS,y1-RADIUS,x1+RADIUS,y1+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x2-RADIUS,y2-RADIUS,x2+RADIUS,y2+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x3-RADIUS,y3-RADIUS,x3+RADIUS,y3+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x4-RADIUS,y4-RADIUS,x4+RADIUS,y4+RADIUS),  fill=255, outline=255)
    vertex_test.save(vertex_path)

def trapezoid(output_path, edge_path, vertex_path):
    #Trapezoid Creation
    image = Image.new("RGB", IMG_SIZE, BACKGROUND)
    points, coordinates = randTrap()
    draw = ImageDraw.Draw(image)
    draw.line(points, fill = get_color(), width = WIDTH)
    image, noise_type = add_noise(image)
    data["noise"].append(noise_type)
    data["coordinates"].append(coordinates)
    image.save(output_path)

    #Edge Test
    edge_test = Image.new("L", IMG_SIZE, 'black')
    draw_edge = ImageDraw.Draw(edge_test)
    draw_edge.line(points, fill = (255), width = (WIDTH + 2*EXTRA_PIX))
    edge_test.save(edge_path)

    #Edge Test
    vertex_test = Image.new("L", IMG_SIZE, 'black')
    draw_vertex = ImageDraw.Draw(vertex_test)
    (x1, y1) = points[0]
    (x2, y2) = points[1]
    (x3, y3) = points[2]
    (x4, y4) = points[3]
    
    draw_vertex.ellipse((x1-RADIUS,y1-RADIUS,x1+RADIUS,y1+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x2-RADIUS,y2-RADIUS,x2+RADIUS,y2+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x3-RADIUS,y3-RADIUS,x3+RADIUS,y3+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x4-RADIUS,y4-RADIUS,x4+RADIUS,y4+RADIUS),  fill=255, outline=255)
    vertex_test.save(vertex_path)

def circle(output_path, edge_path, vertex_path):
    #Circle Creation
    image = Image.new("RGB", IMG_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    points, coordinates = randCirc()
    draw.arc(points, start=0, end=360, fill=get_color(), width = WIDTH)
    image, noise_type = add_noise(image)
    data["noise"].append(noise_type)
    data["coordinates"].append(coordinates)
    image.save(output_path)

    #Edge Test
    edge_test = Image.new("L", IMG_SIZE, 'black')
    draw_edge = ImageDraw.Draw(edge_test)
    (x1, y1) = points[0]
    (x2, y2) = points[1]
    draw_edge.arc((x1-EXTRA_PIX,y1-EXTRA_PIX,x2+EXTRA_PIX,y2+EXTRA_PIX), start = 0, end = 360, fill = (255), width = (WIDTH + 2*EXTRA_PIX))
    edge_test.save(edge_path)

    #Vertex Test
    vertex_test= Image.new("L", IMG_SIZE, 'black')
    vertex_test.save(vertex_path)

def triangle(output_path, edge_path, vertex_path):
    #Triangle Creation
    image = Image.new("RGB", IMG_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    points, coordinates = randTriangle()
    draw.line(points,fill=get_color(), width = WIDTH)
    image, noise_type = add_noise(image)
    data["noise"].append(noise_type)
    data["coordinates"].append(coordinates)
    image.save(output_path)

    #Edge Test
    edge_test = Image.new("L", IMG_SIZE, 'black')
    draw_edge = ImageDraw.Draw(edge_test)
    draw_edge.line(points, fill = (255), width = (WIDTH + 2*EXTRA_PIX))
    edge_test.save(edge_path)

    #Vertex Test
    vertex_test = Image.new("L", IMG_SIZE, 'black')
    draw_vertex = ImageDraw.Draw(vertex_test)
    (x1, y1) = points[0]
    (x2, y2) = points[1]
    (x3, y3) = points[2]
    
    draw_vertex.ellipse((x1-RADIUS,y1-RADIUS,x1+RADIUS,y1+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x2-RADIUS,y2-RADIUS,x2+RADIUS,y2+RADIUS),  fill=255, outline=255)
    draw_vertex.ellipse((x3-RADIUS,y3-RADIUS,x3+RADIUS,y3+RADIUS),  fill=255, outline=255)
    vertex_test.save(vertex_path)
    

def empty(output_path, edge_path, vertex_path):
  #Create empty image
  image = Image.new("RGB", IMG_SIZE, BACKGROUND)
  image, noise_type = add_noise(image)
  coordinates = "None"
  data["noise"].append(noise_type)
  data["coordinates"].append(coordinates)
  image.save(output_path)

  #Edge Test
  edge_test = Image.new("L", IMG_SIZE, 'black')
  edge_test.save(edge_path)

  #Vertex Test
  vertex_test= Image.new("L", IMG_SIZE, 'white')
  vertex_test.save(vertex_path)

#Main Function
if __name__ == "__main__":
  for i in range(N_EXAMPLES):
    
    sq_name = "square " + str(i) + ".bmp"
    sq_edge = "square " + str(i) + " edge.bmp"
    sq_vertex = "square " + str(i) + " vertex.bmp"
    data["file_dir"].append(square_dir)
    data["file_name"].append(sq_name)
    data["class"].append("square")
    square(os.path.join(square_dir, sq_name), os.path.join(test_folder, sq_edge), os.path.join(test_folder, sq_vertex))

    tz_name = "trapezoid " + str(i) + ".bmp"
    tz_edge = "trapezoid " + str(i) + " edge.bmp"
    tz_vertex = "trapezoid " + str(i) + " vertex.bmp"
    data["file_dir"].append(trapezoid_dir)
    data["file_name"].append(tz_name)
    data["class"].append("trapezoid")
    trapezoid(os.path.join(trapezoid_dir, tz_name), os.path.join(test_folder, tz_edge), os.path.join(test_folder, tz_vertex))

    ci_name = "circle " + str(i) + ".bmp"
    ci_edge = "circle " + str(i) + " edge.bmp"
    ci_vertex = "circle " + str(i) + " vertex.bmp"
    data["file_dir"].append(circle_dir)
    data["file_name"].append(ci_name)
    data["class"].append("circle")
    circle(os.path.join(circle_dir, ci_name), os.path.join(test_folder, ci_edge), os.path.join(test_folder, ci_vertex))

    tr_name = "triangle " + str(i) + ".bmp"
    tr_edge = "triangle " + str(i) + " edge.bmp"
    tr_vertex = "triangle " + str(i) + " vertex.bmp"
    data["file_dir"].append(triangle_dir)
    data["file_name"].append(tr_name)
    data["class"].append("triangle")
    triangle(os.path.join(triangle_dir, tr_name), os.path.join(test_folder, tr_edge), os.path.join(test_folder, tr_vertex))

    em_name = "empty " + str(i) + ".bmp"
    em_edge = "empty " + str(i) + " edge.bmp"
    em_vertex = "empty " + str(i) + " vertex.bmp"
    data["file_dir"].append(empty_dir)
    data["file_name"].append(em_name)
    data["class"].append("empty")
    empty(os.path.join(empty_dir, em_name), os.path.join(test_folder, em_edge), os.path.join(test_folder, em_vertex))
    print("Image ", i)
  df = pd.DataFrame(data)
  print(df)
  df.to_csv(csv_file, index=False)
  splitfolders.ratio(data_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1), group_prefix=None)
  try:
    shutil.rmtree(data_folder)
  except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))