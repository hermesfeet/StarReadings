# StarReadings
Neural networks to generate horoscope readings

![alt text](https://pm1.narvii.com/6545/cf449170c00e3f5c777ba5d893a965f042069282_hq.jpg)

## What It Does
This deep neural network looks at a large corpus of horoscope readings to generate a new reading.

## Running the App
You will have to train the model first on your own corpus of horoscope data.  So in bash, cd over to the folder and run to train on a GPU:
> python app/train-model.py

Then, run this file to generate readings:


## Built With / Dependencies
Built in Python 3.6. See the Requirements.txt file.
Mostly regex, Keras, Tensorflow.

## Versioning
Version 1.1.

## Dev Timeline - Next To Do
- Finish training
- Add test system

## License
This project is licensed under the MIT License.

Copyright 2019 - HermesFeet

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Appendix - Corpora to Check Out
- [Word-level neural net Keras/LSTM overview] (https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
- [A Survey of Available Corpora for Building Data-Driven Dialogue Systems - UM Paper] (https://arxiv.org/pdf/1512.05742.pdf)
- [Dialogue Datasets] (https://breakend.github.io/DialogDatasets/)
