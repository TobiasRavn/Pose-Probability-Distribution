# Generating artificial images
## How to run
To create images run
```
blenderproc run create_image.py
```
To display hdf5 file
```
blenderproc vis hdf5 output/0.hdf5
```
To run blenderproc debuger
```
blenderproc debug create_image.py
```
## How to create new object to visulize
Make it in autodesk inventor, choose how the object should look in the top bar. Export as an .obj file which makes an .obj .mtl and a folder with an image in. The .obj describes the object and referes to the .mtl which describes the material which can refere to an image if needed.
