package com.unimelb.yunzhejia.xdt;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.ldps.FP_KNN;
import com.unimelb.yunzhejia.xdt.cl.M_J48;
import com.yunzhejia.cpxc.util.ClassifierGenerator;
import com.yunzhejia.cpxc.util.ClassifierGenerator.ClassifierType;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class ModelEvaluation {
	public static void main(String[] args) throws Exception{
		String[] files = {/*"adult",*/"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		ClassifierType[] types = {ClassifierType.DECISION_TREE, ClassifierType.LOGISTIC, ClassifierType.NAIVE_BAYES, ClassifierType.RANDOM_FOREST, ClassifierType.SVM};
		Double[] rates = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
		ClassifierType[] types = {ClassifierType.RANDOM_FOREST};
//		String[] files = {"anneal","diabetes","labor","vote"};
//		Double[] rates = {0.0, 1.0};
		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
			Instances train = DataUtils.load("data/modified/"+file+"_train.arff");
			Instances test = DataUtils.load("data/modified/"+file+"_test.arff");

			
			AbstractClassifier oracle = ClassifierGenerator.getClassifier(ClassifierType.DECISION_TREE
					);
			oracle.buildClassifier(DataUtils.load("data/original/"+file+"_train.arff"));
			
			for(ClassifierType type:types){
				for(double rate:rates){
//					evalute(oracle,type,train,test,rate, writer,file);
					evaluteFP_KNN(oracle,type,train,test,rate, writer,file);
//					evaluteXDT(oracle,type,train,test,rate, writer,file);
		}}}
		writer.close();
	}
	
	private static void evaluteFP_KNN(AbstractClassifier oracle, ClassifierType type, Instances train, Instances test,
			double percentageOfExplanation, PrintWriter writer, String file) throws Exception {
		Map<Long, Set<Integer>> expls = getExpls(train, oracle, percentageOfExplanation);
		FP_KNN cl= new FP_KNN();
		cl.buildClassifierWithExpl(train,expls);
		
		Evaluation eval = new Evaluation(test);
		
		eval.evaluateModel(cl, test);
		
		System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect() +"  cl="+ type +"  explRate="+percentageOfExplanation);
		writer.println("data ="+ file + "accuracy="+ eval.pctCorrect() + "  cl="+type +"  explRate="+percentageOfExplanation);
		
	}

	public static void evaluteXDT(AbstractClassifier oracle, ClassifierType type, Instances train, Instances test, double percentageOfExplanation, PrintWriter writer, String file) throws Exception{
//		Instances newTrain = modifyDataUsingX(train,oracle,percentageOfExplanation);
//		AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
//		cl.buildClassifier(newTrain);
//		Instances newTrain = modifyDataUsingX(train,oracle,percentageOfExplanation);
		Map<Long, Set<Integer>> expls = getExpls(train, oracle, percentageOfExplanation);
		M_J48 xdt = new M_J48();
//		FP_KNN xdt= new FP_KNN();
//		System.out.println(expls.size());
		xdt.buildClassifierWithExpl(train, expls);
//		xdt.buildClassifier(train);
		Evaluation eval = new Evaluation(test);
		
		eval.evaluateModel(xdt, test);
		
		double avgPrecision = 0;
		double avgRecall = 0;
		double avgF1 = 0;
		int count = 0;
		//explanation quality
		for(Instance ins:test){
			Set<Integer> trueX = ClassifierTruth.getGoldFeature(oracle, ins,0.01);
			Set<Integer> X = DTTruth.getGoldFeature(xdt, ins);
			
			int union = 0;
			for(int a:X){
				if(trueX.contains(a)){
					union++;
				}
			}
			double precision = X.size()==0?0: union/X.size();
			double recall = trueX.size()==0?0:union/trueX.size();
			
			double f1 = 2*(recall*precision)/(recall+precision);
			if(recall ==0 && precision ==0){
				f1=0;
			}
			avgPrecision+=precision;
			avgRecall+=recall;
			avgF1 += f1;
			count++;
		}
		count = test.size();
		avgPrecision /=count;
		avgRecall/=count;
		avgF1/=count;
		
		System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect() +"  cl="+ type +"  explRate="+percentageOfExplanation +" precision="+avgPrecision+" recall="+avgRecall+" f1="+avgF1);
		writer.println("data ="+ file + "accuracy="+ eval.pctCorrect() + "  cl="+type +"  explRate="+percentageOfExplanation +" precision="+avgPrecision+" recall="+avgRecall+" f1="+avgF1);
	}
	
	public static void evalute(AbstractClassifier oracle, ClassifierType type, Instances train, Instances test, double percentageOfExplanation, PrintWriter writer, String file) throws Exception{
		Instances newTrain = modifyDataUsingX(train,oracle,percentageOfExplanation);
		AbstractClassifier cl = ClassifierGenerator.getClassifier(type);
		cl.buildClassifier(newTrain);
		
		Evaluation eval = new Evaluation(test);
		
		eval.evaluateModel(cl, test);
		
		System.out.println("data ="+ file +" accuracy="+ eval.pctCorrect() +"  cl="+ type +"  explRate="+percentageOfExplanation);
		writer.println("data ="+ file + "accuracy="+ eval.pctCorrect() + "  cl="+type +"  explRate="+percentageOfExplanation);
	}
	
	
	private static Map<Long, Set<Integer>> getExpls(Instances train, AbstractClassifier oracle,
			double percentageOfExplanation) throws Exception {
		// TODO Auto-generated method stub
		Map<Long, Set<Integer>>  ret = new HashMap<>();
		for(Instance ins:train){
			Set<Integer> trueX = null;
			trueX = ClassifierTruth.getGoldFeature(oracle, ins, 0.01);
			
			if(Math.random() < percentageOfExplanation){
				ret.put(ins.getID(), trueX);
				ins.setClassValue(oracle.classifyInstance(ins));
			}
		}
//		System.out.println(ret);
		return ret;
	}

	public static Instances modifyDataUsingX(Instances data, AbstractClassifier cl, double percentageOfExplanation) throws Exception{
		Instances ret = new Instances(data);
		for(Instance ins:ret){
			Set<Integer> trueX =  ClassifierTruth.getGoldFeature(cl, ins,0.01);;
			
			if(Math.random() < percentageOfExplanation){
				for(int i =0 ; i < ins.numAttributes()-1;i++){
					if(!trueX.contains(i)){
						ins.setMissing(i);
					}
				}
				ins.setClassValue(cl.classifyInstance(ins));
			}
		}
//		System.out.println(ret);
		return ret;
	}
}
