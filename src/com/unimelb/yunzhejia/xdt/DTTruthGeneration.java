package com.unimelb.yunzhejia.xdt;

import java.util.Random;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class DTTruthGeneration {
	public static void main(String[] args) throws Exception{
		String[] files = {/*"adult",*/"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		String[] files = {"ionosphere"};
		Random rand = new Random(1);
		for(String file:files){
			Instances train = DataUtils.load("data/original/"+file+"_train.arff");
			Instances test = DataUtils.load("data/original/"+file+"_test.arff");
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.DECISION_TREE);
			cl.buildClassifier(train);
			
			
			for(Instance ins:train){
				if(Math.random()<1){
					ins.setClassValue(cl.classifyInstance(ins));
				}else{
					ins.setClassValue(rand.nextInt(train.numClasses()));
				}
			}
			for(Instance ins:test){
				ins.setClassValue(cl.classifyInstance(ins));
			}
			
			DataUtils.save(train, "data/modified/"+file+"_train.arff");
			DataUtils.save(test, "data/modified/"+file+"_test.arff");
		}
	}
}
