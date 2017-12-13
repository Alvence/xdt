package com.unimelb.yunzhejia.xdt;

import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class NormalizeData {
	

	public static void main(String[] args) throws Exception {
		String[] files = {"adult","anneal","balloon","blood","breast-cancer","chess","crx","diabetes","glass","hepatitis","ILPD","ionosphere"
				,"iris","labor","planning","sick","vote"};
		for(String file:files){
			Instances train = DataUtils.load("data/original/"+file+"_train.arff");
			Instances test = DataUtils.load("data/original/"+file+"_test.arff");
			
			Instances dataset = new Instances(train);
			dataset.addAll(test);
			
			Normalize norm = new Normalize();
			norm.setInputFormat(dataset);  // initializing the filter once with training set
			Instances newTrain = Filter.useFilter(train, norm);  // configures the Filter based on train instances and returns filtered instances
			Instances newTest = Filter.useFilter(test, norm);    // create new test set
			
			DataUtils.save(newTrain, "data/norm/"+file+"_train.arff");
			DataUtils.save(newTest, "data/norm/"+file+"_test.arff");
		}
	}

}
