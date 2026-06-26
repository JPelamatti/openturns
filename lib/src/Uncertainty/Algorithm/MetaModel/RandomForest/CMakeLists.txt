#                                               -*- cmake -*-

# Register current directory files
ot_add_current_dir_to_include_dirs ()


ot_add_source_file (RandomForestRegressionAlgorithm.cxx)
ot_add_source_file (RandomForestRegressionResult.cxx)
ot_add_source_file (RandomForestClassificationAlgorithm.cxx)
ot_add_source_file (RandomForestClassificationResult.cxx)
ot_add_source_file (ClassificationRandomForestAlgorithm.cxx)
ot_add_source_file (ClassificationRandomForestResult.cxx)

ot_install_header_file (RandomForestRegressionAlgorithm.hxx)
ot_install_header_file (RandomForestRegressionResult.hxx)
ot_install_header_file (RandomForestClassificationAlgorithm.hxx)
ot_install_header_file (RandomForestClassificationResult.hxx)
ot_install_header_file (ClassificationRandomForestAlgorithm.hxx)
ot_install_header_file (ClassificationRandomForestResult.hxx)
