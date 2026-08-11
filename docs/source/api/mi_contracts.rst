Multiple-Imputation Contracts
=============================

The contracts in this module define validated boundaries for future FCS engines.
Legacy ``impute_mice`` return values remain unchanged, while the opt-in
``impute_mice(..., return_result=True)`` mode returns an immutable
``ImputationResult`` containing the original mask, plan, chain seeds, completed
datasets, and per-iteration histories.

For example:

.. code-block:: python

   from missingly import impute_mice

   result = impute_mice(frame, n_imputations=5, return_result=True)
   completed_datasets = result.data.imputations
   chain_seeds = result.data.plan.seed_sequence

The contract records reproducibility metadata; it does not by itself establish
convergence, MAR validity, or parity with R ``mice``.

.. automodule:: missingly.mi_contracts
   :members:
   :undoc-members:
   :show-inheritance:
