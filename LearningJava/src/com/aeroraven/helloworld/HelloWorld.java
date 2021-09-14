package com.aeroraven.helloworld;

public class HelloWorld {
	public static void main(String[] args) {
		System.out.println("=====Class Constructor=====");
		ClassConstructorDemo x = new ClassConstructorDemo();
		ClassConstructorDemo x2 = new ClassConstructorDemo("ABCDE");
		System.out.println("=====Basic Data Types=====");
		BasicDataTypes x3 = new BasicDataTypes();
		x3.PrintBDTInfo();
	}
}
