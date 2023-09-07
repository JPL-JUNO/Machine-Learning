# the properties and benefits of the different loss functions

.. list-table:: Loss functions
    :widths 30 30 50 50
    :header-rows: 1

    * - Loss function 
      - Use
      - Benefits
      - Disadvantages
    * - L2
      - Regression
      - More stable
      - Less robust
    * - L1
      - Regression
      - More robust
      - Less stable
    * - Pseudo-Huber
      - Regression
      - More robust and stable
      - One more parameter
    * - Hinge
      - Classification
      - Creates a max margin for use in SVM
      - Unbounded loss affected by outliers
    * - Cross-entropy
      - Classification
      - More stable
      - Unbounded loss, less robust