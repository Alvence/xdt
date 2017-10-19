package com.unimelb.yunzhejia.xdt;

import java.util.Set;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class XDT extends AbstractClassifier {
	public class Node{
		boolean isLeaf;
		double label;
		Instances data;
		
		Set<Node> children;
	}
	
	
	private Node root;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		root = new Node();
		
	}

}
