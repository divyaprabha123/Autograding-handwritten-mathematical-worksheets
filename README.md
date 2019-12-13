# Auto-grading of handwritten mathematical worksheets

Digitizing steps of solving a mathematical equation written by freehand on a paper, validating the steps and final answer of the recognized handwritten lines by maintaining the context.

## Workflow

![alt text](https://drive.google.com/uc?id=1UoBwIxsNj4LRQTezyn1KOYrwO9L6gKhJ)

### Prerequisites

Tensorflow

Keras

Opencv


### Example
Each line is corrected separately 

1. Green Box represents - Line is correct
2. Red Box represents - Line is incorrect
![alt-text](https://drive.google.com/uc?id=1-I3WUjVu09SbItEY54xnBy1_00-09jwY)

#### Equation given A * X<sup>2</sup> + B * Y

| Line No       | Equation written     | Expected Ans  |   Actual Ans    |   Status       |
| ------------- |:--------------------:| -------------:| --------------: | --------------:|
| 1      | 56 * 3<sup>2</sup> + 7 * 13 | 595 | 595 | Correct |
| 2      | 56 * 7 + 92 | 595 | 484 | Incorrect |
| 3      | 595 + 92 | 595 | 687 | Incorrect |
| 4      | 595 | 595 | 595 | Correct |
