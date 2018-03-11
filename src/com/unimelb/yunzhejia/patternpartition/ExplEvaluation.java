package com.unimelb.yunzhejia.patternpartition;

import java.util.Map;
import java.util.Set;

import com.yunzhejia.pattern.IPattern;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class ExplEvaluation {
	public static double evalExpl(ExplPartitionWiseLinearModels cl, Instances data, Map<Long, Set<Integer>> expls) throws Exception{
		double ret = 0;
		
		for(int i = 0; i < data.numInstances();i++){
			Instance ins = data.get(i);
			
			cl.m_ReplaceMissingValues.input(ins);
			ins = cl.m_ReplaceMissingValues.output();
		    cl.m_NominalToBinary.input(ins);
		    ins = cl.m_NominalToBinary.output();
			
			
			
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
			 for(int dim = 0; dim < data.numAttributes(); dim++){
				  double ed = (dim == ins.numAttributes()-1? 1 : (expls.get((long)i).contains(dim)?1:0)); 
				 
				  ret += (coe[dim] - ed)*(coe[dim] - ed);
//				  ret += (Math.abs(coe[dim])>cl.the?1:0 - ed)*(Math.abs(coe[dim])>cl.the?1:0 - ed);
			 }
		}
		
		return ret/data.numInstances();
	}
	
	
	public static double precisionExpl(Map<Long, Set<Integer>> expls, Map<Long, Set<Integer>> trueExpls){
		 for(Long id:trueExpls.keySet()){
			 Set<Integer> trueExpl = trueExpls.get(id);
		 }
		 return 0;
	}
}
