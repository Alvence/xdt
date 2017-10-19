package com.unimelb.yunzhejia.xdt;

import java.util.Random;

public class SyntheticDataGenerator {
	
	public static void rules(int N){
		Random random = new Random(1);
		int numFeature = 3;
		for(int i = 0; i < N; i++){
			int vals[]=new int[numFeature];
			int c = 0;
			for(int j = 0; j < numFeature; j++){
				vals[0] = random.nextInt(10);
				vals[1] = random.nextInt(10);
				if(vals[0]>5){
					c = 1;
				}else if (vals[1]>5){
					c = 1;
				}else{
					c = 0;
				}
			}
			if (c == 1){
				vals[2] = random.nextInt(5)+5;
			}else{
				vals[2] = random.nextInt(5);
			}
			for(int j = 0; j < numFeature-1; j++){
				System.out.print(vals[j]+",");
			}
			if(random.nextInt(10)<5){
				System.out.print(vals[numFeature-1]+",");
			}else{
				System.out.print("?,");
			}
			System.out.println(c);
		}
		
	}
	
	public static void ruletest(int N){
		Random random = new Random(1);
		int numFeature = 3;
		for(int i = 0; i < N; i++){
			int vals[]=new int[numFeature];
			int c = 0;
			for(int j = 0; j < numFeature; j++){
				vals[0] = random.nextInt(10);
				vals[1] = random.nextInt(10);
				vals[2] = random.nextInt(10);
				if(vals[0]>5){
					c = 1;
				}else if (vals[1]>5){
					c = 1;
				}else{
					c = 0;
				}
			}
			
			for(int j = 0; j < numFeature; j++){
				System.out.print(vals[j]+",");
			}
			System.out.println(c);
		}
		
	}

	
	public static void main(String[] args) {
//		LOG(1000);
//		DNF2G(200);
//		DNF3G(500);
		rules(100);
		/*try {
			Instances data = DataUtils.load("data/synthetic3.arff");
			List<Instances> datas = new ArrayList<>();
	    	for(int i = 0; i < data.numClasses();i++){
	    		datas.add(new Instances(data,0));
	    	}
	    	
	    	for (Instance ins: data){
	    		if(ins.stringValue(0).equals("2"))
	    			continue;
	    		int index = (int)ins.classValue();
	    		datas.get(index).add(ins);
	    	}
	    	
	    	ScatterPlotDemo3.render(ScatterPlotDemo3.createChart(datas, 1, 2));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
	}

}
