MODEL    := WRN28
VERSIONS := standard robust

# ----------------
# Top-level targets
# ----------------

all: corevectors peepholes eval

corevectors: $(foreach v, $(VERSIONS), corevectors_$(v))

peepholes: $(foreach v, $(VERSIONS), peepholes_$(v))

eval: $(foreach v, $(VERSIONS), modelAcc_$(v) modelRobustness_$(v)) \
      coverageComparison clusterComparison

# ----------------
# Corevectors  (xp_svd.py -l linear)
# ----------------

define corevectors_template
corevectors_$(1):
	python corevectors/xp_svd.py -m $(MODEL) -v $(1) -l linear
endef

$(foreach v, $(VERSIONS), $(eval $(call corevectors_template,$(v))))

# ----------------
# Peepholes  (xp_peepholes.py -r svd)
# ----------------

define peepholes_template
peepholes_$(1):
	python peepholes/xp_peepholes.py -m $(MODEL) -v $(1) -r svd
endef

$(foreach v, $(VERSIONS), $(eval $(call peepholes_template,$(v))))

# ----------------
# Eval — per version  (-m MODEL -v VERSION)
# ----------------

define modelAcc_template
modelAcc_$(1):
	python eval/xp_modelAcc.py -m $(MODEL) -v $(1)
endef

define modelRobustness_template
modelRobustness_$(1):
	python eval/xp_modelRobustness.py -m $(MODEL) -v $(1)
endef

$(foreach v, $(VERSIONS), $(eval $(call modelAcc_template,$(v))))
$(foreach v, $(VERSIONS), $(eval $(call modelRobustness_template,$(v))))

# ----------------
# Eval — model-level  (-m MODEL, both versions handled internally)
# ----------------

svdComparison:
	python eval/xp_SVDComparison.py -m $(MODEL)

modelCalib:
	python eval/xp_modelCalib.py -m $(MODEL)

modelConf:
	python eval/xp_modelConf.py -m $(MODEL)

coverageComparison:
	python eval/xp_CoverageComparison.py -m $(MODEL)

clusterComparison:
	python eval/xp_ClusterComparison.py -m $(MODEL)

# ----------------
# Phony
# ----------------

.PHONY: all corevectors peepholes eval \
        $(foreach v, $(VERSIONS), corevectors_$(v) peepholes_$(v) \
                                  modelAcc_$(v) modelRobustness_$(v)) \
        svdComparison modelCalib modelConf coverageComparison clusterComparison
