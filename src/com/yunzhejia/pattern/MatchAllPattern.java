package com.yunzhejia.pattern;

import java.util.Set;

import weka.core.Instance;
import weka.core.Instances;

public class MatchAllPattern implements IPattern {

	@Override
	public boolean match(Instance ins) {
		return true;
	}

	@Override
	public Instances matchingDataSet(Instances data) {
		return data;
	}

	@Override
	public double support(Instances data) {
		return 1;
	}

	@Override
	public double support() {
		return 1;
	}

	@Override
	public double ratio() {
		return 1;
	}

	@Override
	public double lenghth() {
		return 0;
	}

	@Override
	public boolean contrainAttr(int i) {
		return true;
	}

	@Override
	public void setRatio(double r) {
		// TODO Auto-generated method stub

	}

	@Override
	public Set<ICondition> getConditions() {
		return null;
	}

	@Override
	public IPattern conjuction(IPattern p) {
		return p;
	}

	@Override
	public IPattern disjuction(IPattern p) {
		return this;
	}

	@Override
	public boolean subset(IPattern p) {
		return true;
	}
	
	@Override
	public String toString(){
		return "MatchAllPattern";
	}

}
