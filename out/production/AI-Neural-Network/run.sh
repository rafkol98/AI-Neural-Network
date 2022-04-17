# example command for running the mnist example code (you will need to download data.zip from https://github.com/lephong/minet/tree/main/minet/example/mnist first)
#java -cp lib/*:minet/:. minet/example/mnist/MNISTClassifier 123 minet/example/mnist/data/train.txt minet/example/mnist/data/dev.txt minet/example/mnist/data/test.txt

# example commands for running the practical main function
java -cp lib/*:minet/:src:. src/A4Main part1 123 data/part1/train.txt data/part1/dev.txt data/part1/test.txt data/part1/vocab.txt data/part1/classes.txt
java -cp lib/*:minet/:src:. src/A4Main part2 123 data/part1/train.txt data/part1/dev.txt data/part1/test.txt data/part1/vocab.txt data/part1/classes.txt
java -cp lib/*:minet/:src:. src/A4Main part3 123 data/part1/train.txt data/part1/dev.txt data/part1/test.txt data/part1/vocab.txt data/part1/classes.txt
