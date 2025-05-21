.. dropdown:: Distribution Statement

 | # # # This source code is subject to the license referenced at
 | # # # https://github.com/NRLMMD-GEOIPS.

#############################################################
Instructions for setting up repository from template
#############################################################

*IMPORTANT NOTE: In all commands below, replace @package@ with the name that
you want for your plugin package.*

Obtain the template repository
==============================

If you are a member of the NRLMMD-GEOIPS organization
-----------------------------------------------------

You will be able to create a new repository directly from the template_basic_plugin
repository if you have appropriate permissions within the organization / repositories.

Try this first, and if it doesn't work, follow the instructions for an external user.

#. From github web interface, create a new repository
#. Select "Repository Template" NRLMMD-GEOIPS/template_basic_plugin

   * If template_basic_plugin is not an option, use external user instructions below.
#. Do not select "Include all branches"
#. Select an organization / repository name as desired
#. Include informative description
#. Select public, internal, or private as desired.
#. Select "Create repository"

If any of those steps do not work, follow the external user instructions in the next
section.

If you are an external user
---------------------------

#. Clone the template repository and push it to a repo of your own

   * Change to the GeoIPS packages directory (set during GeoIPS install).

     * ``cd $GEOIPS_PACKAGES_DIR``
   * Clone this repo:
     ``git clone https://github.com/NRLMMD-GEOIPS/template_basic_plugin.git``
   * Move to your package name: ``mv template_basic_plugin @package@``
   * Create a git repository somewhere (e.g. github.com)
   * Move into your package repo: ``cd @package@``
   * Set the repository URL to your new repository:
     ``git remote set-url origin <your repo URL>``
   * Push to your new repository: ``git push -u origin main``

Update your new repository with your own package information
============================================================

#. Update package subdirectory from my_package to @package@

   * ``git mv my_package @package@``
   * ``git commit my_package @package@``

#. Update **README.md** with your appropriate package information.

   * ``cd`` to your package
   * Edit README.md
   * Replace @package@ with your actual package name (my_package_name):

     * :%s/@package@/my_package_name/gc
   * Search for '@' within the README and follow the included instructions to
     update appropriately.
   * Remove all lines containing '@'
   * ``git commit README.md``

#. Update pyproject.toml appropriately for new package name

   * Edit pyproject.toml
   * Update @package@ to package name
   * Add any python package dependencies to "install_requires"
   * ``git commit pyproject.toml``

#. Update/add yaml plugins in @package@/plugins/yaml with desired
   functionality.

   * Template/example YAML files included for reference
   * Modify or delete directories / files as appropriate.
   * Add additional plugins directories and Python modules as needed -
     for examples, see:

     * https://github.com/NRLMMD-GEOIPS/geoips/tree/main/geoips/plugins/yaml
     * https://github.com/NRLMMD-GEOIPS/geoips_plugin_example/tree/main/geoips_plugin_example/plugins/yaml

   * ``git commit .``

#. Update/add modules in @package@/plugins/modules with desired
   functionality.

   * Template/example modules included for reference
   * Modify or delete directories / files as appropriate.
   * Add additional plugins directories and Python modules as needed -
     for examples, see:

     * https://github.com/NRLMMD-GEOIPS/geoips/tree/main/geoips/plugins/modules
     * https://github.com/NRLMMD-GEOIPS/geoips_plugin_example/tree/main/geoips_plugin_example/plugins/modules

   * ``git commit .``

#. Update pyproject.toml appropriately for new Python module-based plugins

   * Edit pyproject.toml
   * Add any new interface modules to "entry_points" (every module added in
     the ``plugins/modules`` subdirectory will have an associated entry point
     in pyproject.toml)
   * Add any python package dependencies to "install_requires"
   * ``git commit pyproject.toml``

#. Add individual test scripts in @package@/tests/scripts/\*.sh

   * ``amsr2.tc_clean.89-Test.sh`` is a direct single_source
     example command - this tests a single product for a single data type. You
     do not have to exhaustively test every piece of functionality with direct
     single source calls - but it can be nice to have one or 2 examples for
     reference. Name your test scripts appropriately.
   * ``test_config.yaml`` is called by ``test_config.sh`` to produce output
     for multiple products with a single call.  Testing all products can be
     more efficiently performed using YAML output config testing vs direct
     single source calls.
   * These test scripts provide both examples of how the package is called via
     geoips, as well as a means of ensuring the processing continues to
     function as updates are made to external dependencies.
   * Rename / delete / add test scripts appropriately.
   * ``git commit tests/scripts``

#. Add all test scripts to @package@/tests/test_all.sh

   * Edit tests/test_all.sh
   * This script is called automatically during exhaustive geoips testing -
     requires 0 return.
   * Ensure all functionality included.
   * ``git commit tests/test_all.sh``

#. Add one example test script to README.md, if desired

   * Edit README.md
   * Add one direct test call to last section, "Test @package@ installation"
   * ``git commit README.md``

#. Update docs/source/releases/latest/initial-commit.yaml with description of
   updates / included modules.

   * Edit docs/source/releases/latest/initial-commit.yaml
   * ``git commit docs/source/releases/latest/initial-commit.yaml``

#. Make sure all new and updated files have been commited and pushed

   * ``git commit .``
   * ``git push``

Clean up any remaining template files
=====================================

In case you missed cleaning up any template files during the initial development,
clean them up here.

#. Remove this 'template_instructions.rst' file

   * ``git rm docs/template_instructions.rst``
   * ``git commit docs/template_instructions.rst``
   * ``git push``

#. Now make sure all the original amsr2 template files are removed. You may have already
   removed these, just make sure they're gone!

   * ``git rm tests/scripts/amsr2.global_clean.89-PCT-Using-Product-Defaults.sh``
   * ``git rm tests/scripts/amsr2.tc_clean.89-PCT-Fully-Specified.sh``
   * ``git rm */plugins/modules/algorithms/pmw_89test.py``
   * ``git rm */plugins/yaml/product_defaults/89-PCT-Test.yaml``
   * ``git rm */plugins/yaml/products/amsr2_fully_specified.yaml``
   * ``git rm */plugins/yaml/products/amsr2_using_product_defaults.yaml``

#. Make sure all removed files have been commited and pushed

   * ``git commit .``
   * ``git push``
