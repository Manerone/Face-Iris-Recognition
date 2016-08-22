# CI396-2016-2
Discipline of Topics in Computer Vision, second semester of 2016.
This repository has my solutions to the problems presented in the discipline.
All code is written in python because it would be easer to do the assignments and
for learning purposes.

## Problems
A brief description of each problem

### Problem 01 - Face Identification

This problem is divided in parts, and it uses two databases:

##### Databases
1. [ORL(AT&T)](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
2. [Yale](http://vision.ucsd.edu/content/yale-face-database)

*Both databases will be present inside the folder of the problem*

##### Parts
1. Compute the mean face for each database
2. Compute the first five eigenfaces for each database
3. Implement the Eigenface ~~and Fisherface~~ method...
 1. Use a cross-validation scheme of leave-one-"expression or lighting" for the Yace Face Database. What expression/lighting is the worst in terms of accuracy?
 2. Use a ten-fold cross-validation scheme and report the mean an stand deviation accuracies for the ORL database. Is there a difference statisical significance between the reported values?
4. Provide a runtime analysis for training and testing
