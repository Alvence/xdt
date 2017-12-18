package com.unimelb.yunzhejia.xdt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifierTruth {
	public static Set<Integer> getGoldFeature(AbstractClassifier cl, Instance instance, double delta) throws Exception {
		
		if(cl instanceof J48){
			return DTTruth.getGoldFeature(cl, instance);
		}
		if(cl instanceof Logistic){
			return LRTruth.getGoldFeature(cl, instance);
		}
		
		double pred = cl.classifyInstance(instance);
		double[] probs = cl.distributionForInstance(instance);
		int numAttr = instance.numAttributes();
		Set<Integer> expl = new HashSet<>();
		Instance copy = (Instance)instance.copy();


		for(int i = 0; i < numAttr && i!=copy.classIndex(); i++){
			Instance temp = (Instance)copy.copy();
			temp.setMissing(i);
			if(Math.abs(cl.distributionForInstance(temp)[(int)pred] - probs[(int)pred]) < delta){
				copy.setMissing(i);
			}
		}
		
		for(int i = 0; i < numAttr && i!=copy.classIndex(); i++){
			if(!copy.isMissing(i)){
				expl.add(i);
			}
		}
		return expl;
	}
	
	public static Map<Long, Set<Integer>> readFromFile(String file) throws Exception{
		Map<Long, Set<Integer>> ret = new HashMap<>();
		
		BufferedReader reader = new BufferedReader(new FileReader(new File(file)));
		String line = null;
		while((line = reader.readLine())!=null){
			String[] e = line.split(",");
			Long id = Long.parseLong(e[0].trim());
			Set<Integer> ex = new HashSet<>();
			for(int i = 1; i < e.length; i++){
				ex.add(Integer.parseInt(e[i].trim()));
			}
//			System.out.println(Arrays.toString(e));
//			System.out.println(id+"       "+ex);
			ret.put(id, ex);
		}
		reader.close();
		
		return ret;
	}
	
	
	
public static void main(String[] args) throws Exception {
	readFromFile("data/modified/expl/adult_train.expl");
	/*
		try {
			Instances train = DataUtils.load("data/original/blood_train.arff");
			Instances test = DataUtils.load("data/original/blood_test.arff");
			
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.DECISION_TREE);
			cl.buildClassifier(train);
			System.out.println(cl);
			
			for(Instance instance:test){
				System.out.println("Ins: "+instance);
				System.out.println(getGoldFeature(cl,instance,0.01));
			}
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
*/
	}
}
