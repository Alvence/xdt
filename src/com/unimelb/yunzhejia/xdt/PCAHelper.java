package com.unimelb.yunzhejia.xdt;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.unimelb.yunzhejia.patternpartition.ExplEvaluation;
import com.yunzhejia.cpxc.util.DataUtils;

import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class PCAHelper {

	public static void main(String[] args) throws Exception {
		String[] files = {"sick"};

		for(String file:files){
			Instances train = DataUtils.load("data/modified/"+file+"_train.arff");

			PrincipalComponents ntb = new PrincipalComponents();
			ntb.setInputFormat(train);
			ntb.m_MaxAttributes = 5;
			Instances binaryTrain = Filter.useFilter(train, ntb);
			
			System.out.println(binaryTrain.numAttributes());
			
			Set<Integer> set = getAttr(ntb, 0);
			System.out.println(set);
			for(int item : set){
				System.out.println(train.attribute(item).name());
			}
			
		}
		
		

	}
	
	public static Set<Integer> getAttr(PrincipalComponents pca, int attrIndex){
		double[] coeff_mags;
	    int num_attrs;
	    int[] coeff_inds;
	    double coeff_value;
	    double cumulative =0.0;
	    int numAttsLowerBound;
	    
	    
	    
	    if (pca.m_MaxAttributes > 0) {
	        numAttsLowerBound = pca.m_NumAttribs - pca.m_MaxAttributes;
	      } else {
	        numAttsLowerBound = 0;
	      }
	      if (numAttsLowerBound < 0) {
	        numAttsLowerBound = 0;
	      }
	    int count = 0;
	    Set<Integer> ret = new HashSet<>();
	    
		for (int i = pca.m_NumAttribs - 1; i >= numAttsLowerBound; i--) {
		    if(count!=attrIndex){
		    	count++;
		    	continue;
		    }
			// build array of coefficients
		      coeff_mags = new double[pca.m_NumAttribs];
		      for (int j = 0; j < pca.m_NumAttribs; j++) {
		        coeff_mags[j] = -Math.abs(pca.m_Eigenvectors[j][pca.m_SortedEigens[i]]);
		      }
		      num_attrs = (pca.m_MaxAttrsInName > 0) ? Math.min(pca.m_NumAttribs,
		    		  pca.m_MaxAttrsInName) : pca.m_NumAttribs;

		      // this array contains the sorted indices of the coefficients
		      if (pca.m_NumAttribs > 0) {
		        // if m_maxAttrsInName > 0, sort coefficients by decreasing magnitude
		        coeff_inds = Utils.sort(coeff_mags);
		      } else {
		        // if m_maxAttrsInName <= 0, use all coeffs in original order
		        coeff_inds = new int[pca.m_NumAttribs];
		        for (int j = 0; j < pca.m_NumAttribs; j++) {
		          coeff_inds[j] = j;
		        }
		      }
		      // build final attName string
		      for (int j = 0; j < num_attrs; j++) {
		        coeff_value = pca.m_Eigenvectors[coeff_inds[j]][pca.m_SortedEigens[i]];
		        if (coeff_value >= 0) {
		          ret.add(coeff_inds[j]);
		        }
//		        attName.append(Utils.doubleToString(coeff_value, 5, 3)
//		          + inputFormat.attribute(coeff_inds[j]).name());
		      }

		      cumulative += pca.m_Eigenvalues[pca.m_SortedEigens[i]];

		      if ((cumulative / pca.m_SumOfEigenValues) >= pca.m_CoverVariance) {
		        break;
		      }
		    }
		return ret;
	}

}
