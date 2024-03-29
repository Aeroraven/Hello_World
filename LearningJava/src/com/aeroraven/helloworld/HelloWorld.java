package com.aeroraven.helloworld;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Iterator;
import java.io.*;
import java.lang.ArrayIndexOutOfBoundsException;

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

abstract class Human{
	protected String gender;
	abstract public void setGender();
	public void showGender() {
		System.out.println(gender);
	}
}
class Male extends Human{
	public void setGender() {
		gender = "Male";
	}
}
class Female extends Human{
	public void setGender() {
		gender = "Female";
	}
}


interface Organ{
	public abstract void function();
}

class Hand implements Organ{
	public void function() {
		System.out.println("I can move");
	}
}

class Foot implements Organ{
	public void function() {
		System.out.println("I can run");
	}
}

public class HelloWorld {
	public static void fun1() {
		System.out.println("HelloWorld");
	}
	public static <E> void printArr(E[] x) {
		for(E i:x) {
			System.out.println(i);
		}
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
		System.out.println("=====Abstract Classes & Abstract Methods=====");
		Human x16 = new Male();
		Human x17 = new Female();
		x16.setGender();
		x17.setGender();
		x16.showGender();
		x17.showGender();
		System.out.println("=====Interface=====");
		Hand x18 = new Hand();
		Foot x19 = new Foot();
		x18.function();
		x19.function();
		System.out.println("=====Enumeration=====");
		enum color{
			Red,Green,Blue
		}
		color rd = color.Red;
		System.out.println(rd);
		for(color i:color.values()) {
			System.out.println(i);
			System.out.println(i.ordinal());
		}
		System.out.println("=====ArrayList=====");
		ArrayList<String> x20 = new ArrayList<String>();
		x20.add("Apple");
		x20.add("Peach");
		x20.add("Orange");
		x20.set(0,"Pineapple");
		System.out.println(x20);
		System.out.println(x20.get(1));
		x20.remove(0);
		System.out.println(x20);
		System.out.println(x20.size());
		System.out.println("=====ArrayList - Sorting=====");
		ArrayList<Integer> x21 = new ArrayList<Integer>();
		x21.add(5);
		x21.add(7);
		x21.add(6);
		Collections.sort(x21);
		for(int i:x21) {
			System.out.println(i);
		}
		System.out.println("=====ArrayList - Converting=====");
		Integer x22[]  = new Integer[x21.size()];
		x21.toArray(x22);
		for(Integer i:x22) {
			System.out.println(i);
		}
		System.out.println("=====HashSet=====");
		HashSet<String> x23 = new HashSet<String>();
		x23.add("A");
		x23.add("C");
		x23.add("C");
		x23.add("B");
		System.out.println(x23);
		x23.remove("C");
		System.out.println(x23);
		System.out.println(x23.contains("C"));
		
		System.out.println("=====HashMap=====");
		HashMap<String,String> x24 = new HashMap<String,String>();
		x24.put("Pear", "Fruit");
		x24.put("Apple", "Fruit");
		x24.put("Tomato", "Vegetable");
		x24.put("Carrot","Vegetable");
		System.out.println(x24);
		x24.remove("Carrot");
		System.out.println(x24);
		x24.replace("Pear", "Fruit2");
		for(String i:x24.keySet()) {
			System.out.println("Key="+i+", Val="+x24.get(i));
		}
		System.out.println("=====Iterator=====");
		System.out.println(x20);
		Iterator<String> it=x20.iterator();
		while(it.hasNext()) {
			System.out.println(it.next());
		}
		System.out.println("=====Generics=====");
		Integer x25[]= {1,2,3,4,5};
		String x26[]= {"A","B","C","D","E"};
		HelloWorld.printArr(x25);
		HelloWorld.printArr(x26);
		System.out.println("=====Exception Handle=====");
		Integer x27[]= {1,2,3};
		try {
			System.out.println(x27[3]);
		}catch(ArrayIndexOutOfBoundsException e) {
			System.out.println("Exception!");
		}
		System.out.println("End");
	}
} 
