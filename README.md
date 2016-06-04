# fizz-buzz-pyfunt

[PyFunt](https://github.com/dnlcrl/PyFunt) implementation of [fizz-buzz-tensorflow](https://github.com/joelgrus/fizz-buzz-tensorflow) (see [http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/))

While the author used a single hidden layer (of 100 units) and 10 digits for the binary representation, I used 2 hidden layers and 12 digits, I also train the network for 2000 epochs instead of 10000 and obtain an accuracy of crica 99.6% on the training set and 100% on the validation set (numbers from 0 to 100).

Here is the output on the validation set after training the net.

      array(['fizzbuzz', '1', '2', 'fizz', '4', 'buzz', 'fizz', '7', '8', 'fizz',
             'buzz', '11', 'fizz', '13', '14', 'fizzbuzz', '16', '17', 'fizz',
             '19', 'buzz', 'fizz', '22', '23', 'fizz', 'buzz', '26', 'fizz',
             '28', '29', 'fizzbuzz', '31', '32', 'fizz', '34', 'buzz', 'fizz',
             '37', '38', 'fizz', 'buzz', '41', 'fizz', '43', '44', 'fizzbuzz',
             '46', '47', 'fizz', '49', 'buzz', 'fizz', '52', '53', 'fizz',
             'buzz', '56', 'fizz', '58', '59', 'fizzbuzz', '61', '62', 'fizz',
             '64', 'buzz', 'fizz', '67', '68', 'fizz', 'buzz', '71', 'fizz',
             '73', '74', 'fizzbuzz', '76', '77', 'fizz', '79', 'buzz', 'fizz',
             '82', '83', 'fizz', 'buzz', '86', 'fizz', '88', '89', 'fizzbuzz',
             '91', '92', 'fizz', '94', 'buzz', 'fizz', '97', '98', 'fizz', 'buzz'],
            dtype='|S8')

