package com.unimelb.yunzhejia.xdt;

import java.io.File;
import java.io.PrintStream;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class DTTruthGeneration {
	public static void main(String[] args) throws Exception{
		String[] files = {"adult","anneal","balloon","blood","breast-cancer","chess","crx","diabetes","glass","hepatitis","ILPD","ionosphere"
				,"iris","labor","planning","sick","vote"};
//		String[] files = {/*"adult",*/"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		String[] files = {"ILPD"};
		Random rand = new Random(1);
		for(String file:files){
			
			Instances train = DataUtils.load("data/norm/"+file+"_train.arff");
			Instances test = DataUtils.load("data/norm/"+file+"_test.arff");
			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.DECISION_TREE);
			cl.buildClassifier(train);
			PrintStream printer = new PrintStream(new File("data/modified/expl/"+file+"_train.expl"));
			for(Instance ins:train){
				if(Math.random()<1){
					ins.setClassValue(cl.classifyInstance(ins));
				}else{
					ins.setClassValue(rand.nextInt(train.numClasses()));
				}
				printer.print(ins.getID()+",");
				Set<Integer> expl = DTTruth.getGoldFeature(cl, ins);
				
				printer.println(expl.toString().replace('[', '\0').replace(']', '\0'));
			}
			printer.close();
			
			printer = new PrintStream(new File("data/modified/expl/"+file+"_test.expl"));
			for(Instance ins:test){
				ins.setClassValue(cl.classifyInstance(ins));
				Set<Integer> expl = DTTruth.getGoldFeature(cl, ins);
				printer.print(ins.getID()+",");
				printer.println(expl.toString().replace('[', '\0').replace(']', '\0'));
			}
			printer.close();
			
			DataUtils.save(train, "data/modified/"+file+"_train.arff");
			DataUtils.save(test, "data/modified/"+file+"_test.arff");
		}
	}
}
