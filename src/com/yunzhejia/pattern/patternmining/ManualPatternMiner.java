package com.yunzhejia.pattern.patternmining;

import java.util.HashSet;
import java.util.Set;

import com.yunzhejia.pattern.ICondition;
import com.yunzhejia.pattern.IPattern;
import com.yunzhejia.pattern.NumericCondition;
import com.yunzhejia.pattern.Pattern;
import com.yunzhejia.pattern.PatternSet;

import weka.core.Instances;

public class ManualPatternMiner implements IPatternMiner {

	public ManualPatternMiner() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp) {
		PatternSet ps = new PatternSet();
		Set<ICondition> cond1 = new HashSet<>();
		cond1.add(new NumericCondition("x",0, 0, 5));
		cond1.add(new NumericCondition("y",1, 5, 10));
		IPattern p1 = new Pattern(cond1);
		
		
		cond1 = new HashSet<>();
		cond1.add(new NumericCondition("x",0, 5, 10));
		cond1.add(new NumericCondition("y",1, 0, 5));
		IPattern p2 = new Pattern(cond1);
		
		
		ICondition condition = new NumericCondition("x",0, 0, 10);
		IPattern p3 = new Pattern(condition);
		
		
		ps.add(p1);
		ps.add(p2);
		ps.add(p3);
		return ps;
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, int featureId) throws Exception {
		throw new Exception("Unsupport operation");
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex) throws Exception {
		throw new Exception("Unsupport operation");
	}

	@Override
	public PatternSet minePattern(Instances data, double minSupp, double minRatio, int classIndex, boolean flag)
			throws Exception {
		throw new Exception("Unsupport operation");
	}

}
