package com.unimelb.yunzhejia.patternpartition;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.xdt.ClassifierTruth;
import com.unimelb.yunzhejia.xdt.PCAHelper;
import com.yunzhejia.pattern.IPattern;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class ExplEvaluation {
	public static double delta = 0.05;
	
	public static double evalExpl(ExplPartitionWiseLinearModels cl, Instances data, Map<Long, Set<Integer>> trueExpls) throws Exception{
		Map<Long, Set<Integer>> expls = new HashMap<>();
		for(int i = 0; i < data.numInstances();i++){
			if(!trueExpls.containsKey((long)i)){
				continue;
			}
			Instance ins = data.get(i);
			
			cl.m_ReplaceMissingValues.input(ins);
			ins = cl.m_ReplaceMissingValues.output();
		    cl.m_NominalToBinary.input(ins);
		    ins = cl.m_NominalToBinary.output();
			
			
			
			 double[] coe = new double[data.numAttributes()];
			 double temp = 0;
			 for(int dim = 0; dim < data.numAttributes(); dim++){
				 coe[dim] = 0;
					for(IPattern p : cl.patterns){
						if(p.match(ins)){
							coe[dim] += cl.A.get(p)[dim];
							temp+=coe[dim];
						}
					}
			 }
			 if(temp>1e-6){
				 Utils.normalize(coe);
			 }
			/* for(int dim = 0; dim < data.numAttributes(); dim++){
				  double ed = (dim == ins.numAttributes()-1? 1 : (expls.get((long)i).contains(dim)?1:0)); 
				 
				  ret += (coe[dim] - ed)*(coe[dim] - ed);
//				  ret += (Math.abs(coe[dim])>cl.the?1:0 - ed)*(Math.abs(coe[dim])>cl.the?1:0 - ed);
			 }*/
			 Set<Integer> expl = new HashSet<>();
			 for(int j = 0; j < coe.length; j++){
				 if(Math.abs(coe[j])>delta){
					 expl.add(j);
				 }
			 }
			 expls.put((long)i, expl);
		}
		return f1Expl(expls,trueExpls);
//		return precisionExpl(expls,trueExpls);
	}
	
	public static double evalExpl(Logistic cl, Instances data, Map<Long, Set<Integer>> trueExpls) throws Exception{
		Map<Long, Set<Integer>> expls = new HashMap<>();
		for(int i = 0; i < data.numInstances(); i++){
			Set<Integer> expl = ClassifierTruth.getGoldFeature(cl, data.get(i), 0.01);
			expls.put((long) i, expl);
		}
		return f1Expl(expls,trueExpls);
//		return precisionExpl(expls,trueExpls);
	}
	
	public static double evalExpl(Logistic cl, PrincipalComponents pca, Instances data, Map<Long, Set<Integer>> trueExpls) throws Exception{
		Map<Long, Set<Integer>> expls = new HashMap<>();
		for(int i = 0; i < data.numInstances(); i++){
			Set<Integer> explpca = ClassifierTruth.getGoldFeature(cl, data.get(i), 0.01);
			Set<Integer> expl = new HashSet<>();
			for(int item : explpca){
				expl.addAll(PCAHelper.getAttr(pca, item));
			}
			expls.put((long) i, expl);
		}
		return f1Expl(expls,trueExpls);
//		return precisionExpl(expls,trueExpls);
	}
	
	
	public static double precisionExpl(Map<Long, Set<Integer>> expls, Map<Long, Set<Integer>> trueExpls){
		double stats = 0;
		int count = 0;
//		System.out.println(expls +"    "+trueExpls);
		for(Long id:trueExpls.keySet()){
			 Set<Integer> trueExpl = trueExpls.get(id);
			 if(expls.containsKey(id)){
				 Set<Integer> expl = expls.get(id);
				 int tp = 0;
				 for(int item:expl){
					 if(trueExpl.contains(item)){
						 tp++;
					 }
				 }
				 double stat = expl.size()==0?0: tp*1.0/expl.size();
				 count++;
				 stats += stat;
			 }
			 
		 }
		 return stats/count;
	}
	
	public static double recallExpl(Map<Long, Set<Integer>> expls, Map<Long, Set<Integer>> trueExpls){
		double stats = 0;
		int count = 0;
		for(Long id:trueExpls.keySet()){
			 Set<Integer> trueExpl = trueExpls.get(id);
			 if(expls.containsKey(id)){
				 Set<Integer> expl = expls.get(id);
				 int tp = 0;
				 for(int item:expl){
					 if(trueExpl.contains(item)){
						 tp++;
					 }
				 }
				 double stat = trueExpl.size()==0?0: tp*1.0/trueExpl.size();
				 count++;
				 stats += stat;
			 }
			 
		 }
		 return stats/count;
	}
	
	
	public static double f1Expl(Map<Long, Set<Integer>> expls, Map<Long, Set<Integer>> trueExpls){
		double stats = 0;
		int count = 0;
		for(Long id:trueExpls.keySet()){
			 Set<Integer> trueExpl = trueExpls.get(id);
			 if(expls.containsKey(id)){
				 Set<Integer> expl = expls.get(id);
				 int tp = 0;
				 for(int item:expl){
					 if(trueExpl.contains(item)){
						 tp++;
					 }
				 }
				 double pre = expl.size()==0?0:tp*1.0/expl.size();
				 double recall = trueExpl.size()==0?0:tp*1.0/trueExpl.size();
				 
				 
				 double stat=0;
				 if(pre+recall!=0){
				   stat= 2*pre*recall/(pre+recall);
				 }
				 count++;
				 stats += stat;
			 }
			 
		 }
		 return stats/count;
	}

	public static double evalExpl(PartitionWiseLinearModels cl, Instances data, Map<Long, Set<Integer>> trueExpls) {
		Map<Long, Set<Integer>> expls = new HashMap<>();
		for(int i = 0; i < data.numInstances();i++){
			if(!trueExpls.containsKey((long)i)){
				continue;
			}
			Instance ins = data.get(i);
			
			
			
			
			 double[] coe = new double[data.numAttributes()];
			 for(int dim = 0; dim < data.numAttributes(); dim++){
				 coe[dim] = 0;
					for(IPattern p : cl.patterns){
						if(p.match(ins)){
							coe[dim] += cl.A.get(p)[dim];
						}
					}
			 }
			 Utils.normalize(coe);
			/* for(int dim = 0; dim < data.numAttributes(); dim++){
				  double ed = (dim == ins.numAttributes()-1? 1 : (expls.get((long)i).contains(dim)?1:0)); 
				 
				  ret += (coe[dim] - ed)*(coe[dim] - ed);
//				  ret += (Math.abs(coe[dim])>cl.the?1:0 - ed)*(Math.abs(coe[dim])>cl.the?1:0 - ed);
			 }*/
			 Set<Integer> expl = new HashSet<>();
			 for(int j = 0; j < coe.length; j++){
				 if(Math.abs(coe[j])>delta){
					 expl.add(j);
				 }
			 }
			 expls.put((long)i, expl);
		}
		return f1Expl(expls,trueExpls);
	}
	
}
