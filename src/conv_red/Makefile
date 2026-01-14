# ----------------
#  Arguments
# ----------------
DATA_DIR := $(or $(DATA_DIR), ../../data)

# ----------------
# Definitions
# ----------------
models := vgg mobilenet
reductions := kernel toeplitz avgpooling
analyses := macs dmd

# ----------------
# Parsing names 
# ----------------
datasets := $(foreach ds, $(models), $(DATA_DIR)/$(ds)/datasets)
corevectors := $(foreach p, $(foreach ds, $(models), $(DATA_DIR)/$(ds)/corevectors), $(foreach r, $(reductions), $(p)/$(r)))
peepholes := $(foreach pp, $(foreach p, $(foreach ds, $(models), $(DATA_DIR)/$(ds)/peepholes), $(foreach r, $(reductions), $(p)/$(r))), $(foreach a, $(analyses), $(pp)/$(a)))

# ----------------
#  Rules
# ----------------
all:
	@echo "hello world"

# ----------------
# Accessible Rules
# ----------------
xp_datasets: $(datasets) 

xp_corevectors: $(corevectors) 

xp_peepholes: $(peepholes) 
	@echo $(peepholes)

# ----------------
# Actual Rules
# ----------------

#  Datasets
$(DATA_DIR)/%/datasets:
	python datasets/xp_datasets.py -m $* -d $(DATA_DIR)

# Corevectors
define cvs_template =
$(1):
	python corevectors/xp_corevectors.py -m $(2) -r $(3) -d $(DATA_DIR)
endef
$(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call cvs_template, $(DATA_DIR)/$(m)/corevectors/$(r), $(m), $(r))) ) )

# Peepholes
define phs_template =
$(1):
	python peepholes/xp_peepholes.py -m $(2) -r $(3) -a $(4) -d $(DATA_DIR)
endef
$(foreach a, $(analyses), $(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call phs_template, $(DATA_DIR)/$(m)/peepholes/$(r)/$(a), $(m), $(r), $(a))))))


