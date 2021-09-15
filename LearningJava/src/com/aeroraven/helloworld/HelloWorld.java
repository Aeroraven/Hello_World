package com.aeroraven.helloworld;

class Test1{
	public static void fun1() {
		final double pi = 3.14159;
		System.out.println(pi);
	}
	public void fun2() {
		System.out.println("fun2");
	}
	public static void fun3(double ...a) {
		double x=0;
		for(double i:a) {
			if(i>x) {
				x=i;
			}
		}
		System.out.println(x);
	}
}

class TestRef{
	public int a;
	public TestRef(int x) {
		a=x;
	}
	protected void finalize() {
		System.out.println("TR:finalize");
	}
}

class TestRef2{
	public static void fun2(TestRef x) {
		x.a++;
	}
	public static void fun2(String x) {
		x = x+x;
	}
}

class Animal{
	protected String ani_name;
	protected String class_name;
	public Animal(String _name) {
		ani_name = _name;
		class_name = "Animal";
	}
	public void show() {
		System.out.println("Name:"+ani_name);
	}
	public void show2() {
		System.out.println("Name:"+ani_name);
		System.out.println("Class:"+class_name);
	}
	public void bite() {
		System.out.println(ani_name+"不会咬人");
	}
}

class Cat extends Animal{
	public Cat(String _name) {
		super(_name);
		super.class_name = "Dog";
	}
	public void bite() {
		System.out.println(ani_name+"会咬人");
	}
	public void bite2() {
		super.bite();
		this.bite();
	}
}

public class HelloWorld {
	public static void fun1() {
		System.out.println("HelloWorld");
	}
	public static void main(String[] args) {
		System.out.println("=====Class Constructor=====");
		ClassConstructorDemo x = new ClassConstructorDemo();
		ClassConstructorDemo x2 = new ClassConstructorDemo("ABCDE");
		System.out.println("=====Basic Data Types=====");
		BasicDataTypes x3 = new BasicDataTypes();
		x3.PrintBDTInfo();
		System.out.println("=====Class=====");
		HelloWorld.fun1();
		Test1 x4 = new Test1();
		Test1.fun1();
		x4.fun2();
		x4.fun1();
		System.out.println("=====Reference=====");
		TestRef x5 = new TestRef(3);
		TestRef x6 = x5;
		x6.a = 4;
		System.out.println(x5.a);
		System.out.println("=====Instance of=====");
		System.out.println(x5 instanceof TestRef);
		System.out.println(x6 instanceof Object);
		System.out.println("=====Loop=====");
		int a[]= {1,3,5,7,9};
		for(int i:a) {
			System.out.println(i);
		}
		System.out.println("=====Argument Reference=====");
		TestRef x7 = new TestRef(3);
		String x8 = "abcde";
		TestRef2.fun2(x7);
		System.out.println(x7.a);
		TestRef2.fun2(x8);
		System.out.println(x8);
		System.out.println("=====Methods=====");
		Test1.fun3(3.14,12.4,10.12,0.4);
		System.out.println("=====Finalize=====");
		x7 = null;
		System.gc();
		Cat x7b = new Cat("Dog");
		System.out.println("=====Inherit=====");
		Cat x9 = new Cat("Dog");
		x9.show();
		System.out.println("=====Super & This=====");
		Cat x10 = new Cat("Dog");
		Animal x11 = new Animal("Cat");
		Animal x12 = new Cat("Rabbit");
		x10.show2();
		x11.show2();
		x12.show2();
		System.out.println("=====Overriding=====");
		Animal x13 = new Cat("老鼠");
		Animal x14 = new Animal("兔子");
		Cat x15 = new Cat("狗");
		x13.bite();
		x14.bite();
		x15.bite2();
	}
} 
