rm -f minet/*/*.class minet/*/*/*/.class src/*.class
javac -cp lib/*:minet/:src:. minet/*/*.java minet/*/*/*.java src/*.java
