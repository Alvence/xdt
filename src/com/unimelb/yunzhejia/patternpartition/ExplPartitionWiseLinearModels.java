package com.unimelb.yunzhejia.patternpartition;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.yunzhejia.cpxc.util.DataUtils;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.MatchAllPattern;
import com.yunzhejia.pattern.PatternSet;
import com.yunzhejia.pattern.patternmining.IPatternMiner;
import com.yunzhejia.pattern.patternmining.RFPatternMiner;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ExplPartitionWiseLinearModels extends AbstractClassifier {

	Instances train;
	
	PatternSet patterns;
	
	Map<IPattern, double[]> A;
	
	
	double minSupp = 0.1;
	double minRatio = 3;
	int defaultClass=-1;
	
	//delta - importance
	double delta = 1e-2;
	
	/** The filter used to make attributes numeric. */
	public NominalToBinary m_NominalToBinary;

	  /** The filter used to get rid of missing values. */
	public ReplaceMissingValues m_ReplaceMissingValues;
	Random rand = new Random(0);
	
	int T = 1000;
	@Override
	public void buildClassifier(Instances data) throws Exception {
		buildClassifierWithExpl(new RFPatternMiner(),data,null);
	}
	
	public void buildClassifierWithExpl( IPatternMiner pm ,Instances train, Map<Long, Set<Integer>> expls) throws Exception {
//		double gammas[] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
		double lambdas[] = {0.001};
		double steps[] = {0.01,0.1};
		
		double gamma = 0.5;
		
		double bestStep = 0.0;
		double bestLambda = 0.0;
		double bestObj = 0.0;
		
		Instances temp = new Instances(train);
		
		Instances valid = new Instances(temp,0);
		Instances test = new Instances(temp,0);
		Map<Long, Set<Integer>> validExpl = new HashMap<>();
		Map<Long, Set<Integer>> testExpl = new HashMap<>();

		
		int pivot = (int)(temp.numInstances()*0.7);
		for(int i = 0; i < temp.numInstances(); i++){
			if( i < pivot){
				valid.add(temp.get(i));
				if(expls.containsKey((long)i)){
					validExpl.put((long)i, expls.get((long)i));
				}
			}else{
				test.add(temp.get(i));
				if(expls.containsKey((long)i)){
					testExpl.put((long)(i-pivot), expls.get((long)i));
				}
			}
		}
		
		for(double step:steps){
			for(double lambda:lambdas){
				ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
				cl.buildClassifierWithExpl(pm, valid, validExpl, gamma, lambda, step);
				
				Evaluation eval = new Evaluation(test);
				
//				cl.buildClassifier(train);
//				cl.buildClassifxierWithExpl(train, expls);
				eval.evaluateModel(cl, test);
				double losX = ExplEvaluation.evalExpl(cl,test,testExpl);
				double obj = eval.correct()/100.0 + gamma * losX;
				if(obj > bestObj){
					bestStep = step;
					bestLambda = lambda;
					bestObj = obj;
				}
			}
		}
		
		System.out.println("lambda="+bestLambda+"   step="+bestStep);
		buildClassifierWithExpl(pm,train,expls,gamma, bestLambda, bestStep);
	}
	
	public void buildClassifierWithExpl( IPatternMiner pm ,Instances train, Map<Long, Set<Integer>> expls, double gamma, double lambdaP, double stepSize) throws Exception {
		 Instances instances = new Instances(train);
		 instances.deleteWithMissingClass();
		 
		 
		 double beta1=0.9;
		 double beta2=0.999;
		 double eps = 1e-8;
		 double m0 = 0;
		 double v0 = 0;
		 double m1 = 0;
		 double v1 = 0;
		 
		// Replace missing values
		 m_ReplaceMissingValues = new ReplaceMissingValues();
		 m_ReplaceMissingValues.setInputFormat(instances);
		 instances = Filter.useFilter(instances, m_ReplaceMissingValues);

		

		 // Transform attributes
		 m_NominalToBinary = new NominalToBinary();
		 m_NominalToBinary.setInputFormat(instances);
		 instances = Filter.useFilter(instances, m_NominalToBinary);
		 
		 
//		 pm = new ManualPatternMiner();
//		 IPatternMiner pm = new RFPatternMiner();
//		 pm = new ParallelCoordinatesMiner();
//		 IPatternMiner pm = new GcGrowthPatternMiner(discretizer);
		 patterns = pm.minePattern(instances, minSupp);
//		 patterns = new PatternSet();
//		 patterns.clear();
		 //make a global pattern that matches all instances
		 IPattern globalPattern = new MatchAllPattern();
		 patterns.add(globalPattern);
		 
//		 System.out.println(patterns);
//		 System.out.println(patterns.get(0).matchingDataSet(instances));
//		 System.out.println(patterns.get(1).matchingDataSet(instances));
//		 System.out.println(patterns.get(2).matchingDataSet(instances));
//		 if(true)
//		 return;
		 //initialize A
		 A = new HashMap<>();
		 for(IPattern p:patterns){
//			 Instances mds = p.matchingDataSet(instances);
//			 LinearRegression lr = new LinearRegression();
//			 lr.buildClassifier(mds);
//			 A.put(p, lr.coefficients());
			 double[] iniCoe = new double[instances.numAttributes()];
			 for(int i = 0; i < iniCoe.length; i++){
//				 iniCoe[i]=(int)(Math.random()*10+1);
				 iniCoe[i]=rand.nextDouble();
			 }
//			 System.out.println(p);
			 A.put(p, iniCoe);
		 }
	
		 double[] objs = new double[20];
		 int obj_index = 0;
		 
		 double oldObj = Double.MAX_VALUE;
		 double newObj = Double.MAX_VALUE;
		 
		 //iteration
		 for(int t = 1; t < T ; t++){
//			 System.out.println("t="+t);
//			 showStat();
			 
			double[] preds = new double[instances.numInstances()];
			
			//calc current accuracy
			 
			int correct = 0;
			double explCorr = 0.0;
			 for (int i = 0; i < instances.numInstances();i++){
				 Instance ins = instances.get(i);
				 //calculate errors
				 double[] coe = new double[instances.numAttributes()];
				 for(int d = 0; d < coe.length; d++){
						coe[d] = 0;
						for(IPattern p : patterns){
							if(p.match(ins)){
								coe[d] += A.get(p)[d];
							}
						}
				 }

				 double pred = 0;
				 for(int d = 0; d < coe.length; d++){
					 pred+= coe[d] * (d == ins.numAttributes()-1? 1:ins.value(d));
				 }
				 preds[i] = pred;
				 
				 //calc expl loss
				 if(expls!=null && expls.get((long)i)!=null){
					 int explCount = 0;
					 Set<Integer> expl = expls.get((long)i);
					 for(int index = 0; index<coe.length-1; index++){
						 double item = coe[index];
						 if(Math.abs(item)>=delta && expl.contains(index)){
							 explCount++;
						 } else if(Math.abs(item)<delta && !expl.contains(index)){
							 explCount++;
						 }
					 }
					 explCorr += explCount*1.0/(coe.length-1);
				 }
				 
				 int c = pred>0?1:0;
				 if(c == ins.classValue()){
					 correct++;
				 }
			 }
			 double acc = correct*1.0/instances.numInstances();
			 double losX = explCorr*1.0/expls.size();
			 
			 
//			 System.out.println("acc="+acc+"   losX="+losX);
			 objs[obj_index%20] = acc+ gamma*losX;
			 obj_index++;
			 
			 //terminate condition
			 if(terminate(objs)){
//				 System.out.println(Arrays.toString(objs));
//				 System.out.println("T="+t);
//				 break; 
			 }
			 
			 
			
			 
			 
			 
//			 System.out.println("preds:  "+Arrays.toString(preds));
			 
			 Map<IPattern, double[]> At = new HashMap<>();
			 
			 
			 for(IPattern p : patterns){
				 double[] coet = new double[instances.numAttributes()];
				 double[] coe = new double[instances.numAttributes()];
				 double[] coe_current = new double[instances.numAttributes()];
				 for(int k = 0; k < coe_current.length; k++){
					 coe_current[k] = A.get(p)[k];
					 coe[k] = A.get(p)[k];
				 }
				//M1 optimization
				 double theta = 0.0;
				 double total = 0.0;
				 int count = 0;
				 Arrays.sort(coe_current);
				 for(int i = coe_current.length-1; i>=0; i--){
					 double pivot = coe_current[i];
					 double sum = 0;
					 for(int j= coe_current.length-1; j > i; j--){
						 sum+= (Math.abs(coe_current[j])-Math.abs(coe_current[i]));
					 }
					 if(sum < lambdaP){
						 count++;;
						 total+= Math.abs(pivot);
					 }else{
						 break;
					 }
				 }
				 theta = (total-lambdaP)/count;
				 theta = (theta>0? theta:0);
				 
				 
				 
				 for(int d = 0; d < coet.length; d++){
					
					 double der = 0;
					 
					 double der2 = 0;
					 
					 double reg = 0;
					 for (int i = 0; i < instances.numInstances();i++){
						 Instance ins = instances.get(i);
						 
//						 for(int dim = 0; dim < coe.length; dim++){
//								coe[dim] = 0;
//								for(IPattern pa : patterns){
//									if(pa.match(ins)){
//										coe[dim] += A.get(pa)[dim];
//									}
//								}
//						 }
//
//						 double pred = 0;
//						 for(int dim = 0; dim < coe.length; dim++){
//							 pred+= coe[dim] * (dim == ins.numAttributes()-1? 1:ins.value(dim));
//						 }
//						 preds[i] = pred;
						 
						 
						 double y = ins.classValue();
						 der += (-1)*(y-(1/(1+Math.exp(-1*preds[i]))))  *(p.match(ins)?1:0)*(d == ins.numAttributes()-1? 1:ins.value(d));
					 
//						 double y = ins.classValue() == 0? -1:1;
//						 der += (-y)*(1/(1+Math.exp(-y*preds[i])))*(p.match(ins)?1:0)*(d == ins.numAttributes()-1? 1:ins.value(d));
						 
						 //expl loss
						 if(expls!=null && expls.get((long)i)!=null){
						 if(expls.get((long)i).contains(d) && Math.abs(coe[d])<delta){
							 der2 += (p.match(ins)?1:0);
						 }else if((!expls.get((long)i).contains(d))&& Math.abs(coe[d])>delta){
							 der2 += (coe[d]>0?1:-1)* (p.match(ins)?1:0);
						 }
						 	double e = expls.get((long)i).contains(d)?1:0;
						 	reg += (1-2*e)* (p.match(ins)?1:0);
						 }
					 }
					 
					 if(expls!=null && expls.size()>0){
						 reg = reg*gamma / expls.size();
					 }
					 
					 
					 double gr = 0;
					 
					 der = der/instances.numInstances();
					 
					 if(expls!=null && expls.size()>0){
						 der2 /= expls.size();
					 }
					 
//					 System.out.println(der);
					 coet[d] = A.get(p)[d] - stepSize*der;
					 
					 gr+= der;
					 
					 //loss expl
					 coet[d] = coet[d] - stepSize * gamma* der2;
//					 System.out.println(der);
					 
					 gr+=gamma*der2;
					 
					 /*if (coet[d]>stepSize*reg){
						 coet[d] -= stepSize*reg;
					 }else if(coet[d]<-stepSize*reg){
						 coet[d] += stepSize*reg;
					 }else{
						 coet[d] = 0;
					 }*/
					 
					 
					 
					 
					 
					 if(! (p instanceof MatchAllPattern)){
//						 coet[d] = (coet[d]>0?1:-1)*Math.min(Math.abs(coet[d]),theta);
//						 if (coet[d] > lambda0){
//							 coet[d] -= lambda0;
//						 }else if(coet[d]< - lambda0){
//							 coet[d] += lambda0;
//						 }
//						 coet[d] -= stepSize*lambda0*2*coet[d];
						 
						 if (coet[d]>stepSize*lambdaP){
							 coet[d] -= stepSize*lambdaP;
							 gr += lambdaP;
						 }else if(coet[d]<-stepSize*lambdaP){
							 coet[d] += stepSize*lambdaP;
							 gr -= lambdaP;
						 }else{
							 coet[d] = 0;
						 }
						 
					 }
//					 coet[d] = (coet[d]>0?1:-1)* ((Math.abs(coet[d] - stepSize*lambda0)>0? Math.abs(coet[d] - stepSize*lambda0):0 ));
					 
					 
					 m1 = beta1 * m0 + (1-beta1)*gr;
					 v1 = beta2 * v0 + (1-beta2)*gr*gr;
					 
					 m0 = m1;
					 v0 = v1;
					 
					 double mt = m1 /(1-beta1);
					 double vt = v1/(1-beta2);
					 
//					 coet[d] = A.get(p)[d] - stepSize/(Math.sqrt(vt)+eps)*mt;
					 
					 coet[d] = (Math.abs(coet[d])<delta?0:coet[d]);
				 }
				 At.put(p, coet);
			 }
			
			A = At;
		 }
//		 for(IPattern p:patterns){
//			 System.out.println(p+"   "+Arrays.toString( A.get(p)));
//		 }
	}
	
	private boolean terminate(double[] objs) {
//		System.out.println(Arrays.toString(objs));
		double dif = 0;
		for(int i = 0; i < objs.length-1; i++){
			dif = dif+Math.abs(objs[i+1]-objs[i]);
		}
		if(dif<1e-3){
			return true;
		}
		return false;
	}

	public void showStat(){
		for(IPattern p:patterns){
			System.out.println(p+":  " + Arrays.toString(A.get(p)));
		}
	}
	
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		m_ReplaceMissingValues.input(instance);
	    instance = m_ReplaceMissingValues.output();
	    m_NominalToBinary.input(instance);
	    instance = m_NominalToBinary.output();
		
		double[] coe = new double[instance.numAttributes()];
		double result = 0;
		for(int d = 0; d < coe.length; d++){
			coe[d] = 0;
			for(IPattern p : patterns){
				if(p.match(instance)){
					coe[d] += A.get(p)[d];
				}
			}
			result+= coe[d] * (d == instance.numAttributes()-1? 1:instance.value(d));
		}
		
		return result>0?1:0;
	}
/*
	private Instances findNearest(Instance instance, int k, Instances headerInfo) {
		// TODO Auto-generated method stub
		Instances nei = new Instances(headerInfo,0);
		
		double[] distances = new double[train.size()];
		for(int i = 0; i < train.size(); i++){
			for(IPattern p:ps){
				if(p.match(instance)&&p.match(train.get(i))){
					distances[i] += p.support(train);
				}
			}
		}
		
		while(nei.size()<k){
			double large = -1;
			int index = -1;
			for(int i = 0;i < train.size();i++){
				if(distances[i]>large){
					large = distances[i];
					index = i;
				}
			}
			nei.add(train.get(index));
			distances[index] = -1;
		}
		
		
		return nei;
	}
	*/
	
	
	public static void aamain(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//				"labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		String[] files = {"balloon","blood","crx","diabetes","hepatitis", "labor", "sick", "vote"};
		String[] files = {"synthetic_10samples"};
//		String[] files = {"balloon"};
		
		
		IPatternMiner[] pms = {new RFPatternMiner()};//, new ParallelCoordinatesMiner()};
		boolean[] flags = { true};//, false};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
				for(boolean flag:flags){
				for(IPatternMiner pm:pms){
			Instances train = DataUtils.load("data/"+file+"_train.arff");
			Instances test = DataUtils.load("data/"+file+"_train.arff");
//			Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//			Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
			
			Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/"+file+"_train.expl");
			Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/"+file+"_train.expl");
			ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
//			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.LOGISTIC);
			
			Evaluation eval = new Evaluation(test);
			
//			cl.buildClassifier(train);
//			cl.buildClassifxierWithExpl(train, expls);
			cl.buildClassifierWithExpl(pm, train, flag?expls:null);
//			cl.buildClassifierWithExpl(pm, train, explsTest, 0.5, 0.001, 0.1);
			eval.evaluateModel(cl, test);
			double losX = ExplEvaluation.evalExpl(cl,test,explsTest);
			
			System.out.println("data ="+ file +"pm="+pm+" flag="+flag+" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);
			}}
		}
	}
	
	public static void main(String[] args) throws Exception{
//		String[] files = { /*"adult",*/"balloon","blood"/*,"breast-cancer","chess"*/,"crx","diabetes","hepatitis",/*"ionosphere",*/
//				"labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer",/*"chess",*/"crx","diabetes","glass","hepatitis","ionosphere", "labor","sick","vote"};
//		String[] files = {"anneal","balloon","blood","breast-cancer","diabetes","iris","labor","vote"};
//		String[] files = {"balloon","blood","hepatitis", "labor", "vote","breast-cancer","crx","diabetes"};
//		String[] files = {"vote","crx","diabetes","planning","vote"}; //"titanic","sonar","hypo"
		
//		String[] files = {"titanic","sonar","hypo"};
		String[] files = {"balloon","blood","crx","diabetes","hepatitis","hypo", "labor","sick","titanic","vote"};
		
		IPatternMiner[] pms = {new RFPatternMiner()};//, new ParallelCoordinatesMiner()};
		boolean[] flags = { true};//, false};
//		PrintWriter writer = new PrintWriter(new File("tmp/stats.txt"));
		for(String file:files){
				for(boolean flag:flags){
				for(IPatternMiner pm:pms){
			Instances train = DataUtils.load("data/modified/"+file+"_train.arff");
			Instances test = DataUtils.load("data/modified/"+file+"_test.arff");
//			Instances train = DataUtils.load("data/"+"synthetic_10samples.arff");
//			Instances test = DataUtils.load("data/"+"synthetic_10samples.arff");
			
			Map<Long, Set<Integer>> expls = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_train.expl");
			Map<Long, Set<Integer>> explsTest = ClassifierTruth.readFromFile("data/modified/expl/"+file+"_test.expl");
			ExplPartitionWiseLinearModels cl = new ExplPartitionWiseLinearModels();
//			AbstractClassifier cl = ClassifierGenerator.getClassifier(ClassifierGenerator.ClassifierType.LOGISTIC);
			
			Evaluation eval = new Evaluation(test);
			
//			cl.buildClassifier(train);
//			cl.buildClassifierWithExpl(train, expls);
			cl.buildClassifierWithExpl(pm, train, flag?expls:null);//,0.5,0.001,0.01);
			eval.evaluateModel(cl, test);
			double losX = ExplEvaluation.evalExpl(cl,test,explsTest);
			
			System.out.println("1000 data ="+ file +"pm="+pm+" flag="+flag+" accuracy="+ eval.pctCorrect()+"  losExpl="+losX);
			}}
		}
	}
	
}
