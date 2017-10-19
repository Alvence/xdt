/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ModelSelection.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package com.unimelb.yunzhejia.xdt.cl;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;

import weka.core.Instances;
import weka.core.RevisionHandler;

/**
 * Abstract class for model selection criteria.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public abstract class M_ModelSelection
  implements Serializable, RevisionHandler {

  /** for serialization */
  private static final long serialVersionUID = -4850147125096133642L;

  /**
   * Selects a model for the given dataset.
   *
   * @exception Exception if model can't be selected
   */
  public abstract M_ClassifierSplitModel selectModel(Instances data) throws Exception;

  /**
   * Selects a model for the given train data using the given test data
   *
   * @exception Exception if model can't be selected
   */
  public M_ClassifierSplitModel selectModel(Instances train, Instances test) 
       throws Exception {

    throw new Exception("Model selection method not implemented");
  }

public M_ClassifierSplitModel selectModelWithExpl(Instances data, Map<Long, Set<Integer>> expls) throws Exception {
	// TODO Auto-generated method stub
	 throw new Exception("Model selection method not implemented");
}
}
