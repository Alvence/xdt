package com.unimelb.yunzhejia.xdt;

import java.util.HashSet;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.cl.M_C45Split;
import com.unimelb.yunzhejia.xdt.cl.M_ClassifierTree;
import com.unimelb.yunzhejia.xdt.cl.M_J48;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifierTruth {
	public static Set<Integer> getGoldFeature(AbstractClassifier cl, Instance instance, double delta) throws Exception {
		
		if(cl instanceof J48){
			return DTTruth.getGoldFeature(cl, instance);
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
	
	
	
public static void main(String[] args) {
		
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
		

	}
}
