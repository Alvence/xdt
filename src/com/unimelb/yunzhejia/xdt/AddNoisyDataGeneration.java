package com.unimelb.yunzhejia.xdt;

import java.io.File;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class AddNoisyDataGeneration {
	public static void main(String[] args) throws Exception{
		String[] files = {"adult","anneal","balloon","blood","breast-cancer","chess","crx","diabetes"
				,"glass","hepatitis","ILPD","ionosphere","iris","labor","planning","sick","vote"};
//		String[] files = {/*"adult",*/"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		String[] files = {"iris"};
		Random rand = new Random(1);
		double rate = 0.5;
		
		for(String file:files){
			Instances train = DataUtils.load("data/norm/"+file+"_train.arff");
			Instances test = DataUtils.load("data/norm/"+file+"_test.arff");
			PrintStream printer = new PrintStream(new File("data/noisy"+(int)(rate*100)+"/expl/"+file+"_train.expl"));
			
			int numOldFeature = train.numAttributes();
			Set<Integer> expl = new HashSet<>();
			for(int i = 0; i < numOldFeature - 1; i++){
				expl.add(i);
			}
			
			int numberOfNewFeatures = (int)(rate * train.numAttributes());
			for(int i = 0; i < numberOfNewFeatures; i++){
				train.insertAttributeAt(new Attribute("NewNumeric"+i), train.numAttributes()-1);
				test.insertAttributeAt(new Attribute("NewNumeric"+i), test.numAttributes()-1);
//				 Add filter;
//				 filter = new Add();
//			        filter.setAttributeIndex("last");
//			        filter.setAttributeName("NewNumeric");
//			        filter.setInputFormat(train);
//			        train = Filter.useFilter(train, filter);
//			        test = Filter.useFilter(test, filter);
			}
//		 	System.out.println(train.numAttributes());
			int c = 0;
			for(Instance ins:train){
				for(int i = 0; i < numberOfNewFeatures; i++){
					ins.setValue(train.numAttributes() - 2 - i, rand.nextDouble());
				}
				printer.print((c++)+",");
				printer.println(expl.toString().replace('[', ' ').replace(']', ' ').trim());
			}
			printer.close();
			
			printer = new PrintStream(new File("data/noisy"+(int)(rate*100)+"/expl/"+file+"_test.expl"));
			c = 0;
			for(Instance ins:test){
				for(int i = 0; i < numberOfNewFeatures; i++){
					ins.setValue(train.numAttributes() - 2 - i, rand.nextDouble());
				}
				printer.print((c++)+",");
				printer.println(expl.toString().replace('[', ' ').replace(']', ' '));
			}
			printer.close();
//			 System.out.println(train);
			DataUtils.save(train,"data/noisy"+(int)(rate*100)+"/"+file+"_train.arff");
			DataUtils.save(test, "data/noisy" +(int)(rate*100)+"/"+file+"_test.arff");
		}
	}
}
