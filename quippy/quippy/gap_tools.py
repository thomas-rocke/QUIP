import numpy as np
from xml.etree.ElementTree import parse, fromstring, tostring, ElementTree
import re
import os
from quippy.descriptors import Descriptor
from quippy.potential import Potential
from ase.data import chemical_symbols
from quippy.clustering import get_cur_scores

try: # Try to use triangular solve (faster), if available
    from scipy.linalg import solve_triangular as solve
except ImportError: # Revert to general solve
    solve = np.linalg.solve


def ard_se(x, y, x_cut, y_cut, theta):
    N = x.shape[0]
    M = y.shape[0]
    K = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            K[i, j] = np.exp(-(x[i] - y[j])**2 / (2 * theta**2)) * x_cut[i] * y_cut[j]
    return K

def dot_product(x, y, x_cut, y_cut, xi):
    K = ((x @ y.T) ** xi) * (x_cut[:, np.newaxis] @ y_cut[:, np.newaxis].T)
    return K 

# xml covariance_type int -> covariance kernel function
kerns = [None, ard_se, dot_product]


class DescXMLWrapper():
    '''
    Small wrapper for storing key descriptor info from a GAP xml file

    Key attributes:
    self.quip_desc : Corresponding quippy.descriptors.Descriptor object
    self.nsparse : Number of sparse points in the descriptor sparse GP
    self.weights : Weights for the descriptor sparse GP
    self.sparse_cuts : Cutoff function evaluations for each of the sparse points
    self.cov_type : Covariance kernel string
    self.desc_type : Name of descriptor function
    self.name : Useful name for the descriptor


    '''
    _Z_regex = "(Z|z)[1-9]*\s?=\s?([1-9]+)"  # RegEx to search command line for "Z=, Z1 = or z2= style args"
    _Z_regex = re.compile(_Z_regex)

    # RegEx to find the dot_product exponents from command line args
    _exponent_regex = "exponents\s?=\s?.((\s?(-\d+)\s?)+)."
    _exponent_regex = re.compile(_exponent_regex)

    def __init__(self, desc_xml):
        self.weights = np.array([float(child.attrib["alpha"])
                                for child in desc_xml if "sparseX" in child.tag])
        self.sparse_cuts = np.array([float(
            child.attrib["sparseCutoff"]) for child in desc_xml if "sparseX" in child.tag])

        self.nsparse = self.weights.shape[0]

        self.delta = float(desc_xml.attrib["signal_variance"])
        self._int_cov_type = int(desc_xml.attrib["covariance_type"])

        if self._int_cov_type == 1:  # ard_se
            self.cov_prop = float([child.text.split()[0]
                                  for child in desc_xml if "theta" in child.tag][0])
            self.cov_type = "ard_se"
        elif self._int_cov_type == 2:  # dot_product
            self.cov_prop = float(desc_xml.attrib["zeta"])
            self.cov_type = "dot_product"
        else:
            self.cov_prop = []
            self.cov_type = "UNKNOWN"

        _desc_children = [
            child for child in desc_xml if "descriptor" in child.tag]

        self._cmd = _desc_children[0].text

        self.desc_type = self._cmd.split(" ")[0]

        Zs = self._Z_regex.findall(self._cmd)

        self.Zs = [Z[1] for Z in Zs]

        self.name = "".join([chemical_symbols[int(Z)]
                            for Z in self.Zs]) + " " + self.desc_type

        self.quip_desc = Descriptor(self._cmd)
        self.cov_func = kerns[self._int_cov_type]

        self.sparseX = None

    def set_sparsex(self, sparsex):
        self.sparseX = sparsex

    def get_energy_design(self, structures):
        '''
        Gets the segment of a design matrix corresponding to an energy observation on the structure for this descriptor
        ''' 
        from ase.atoms import Atoms

        if type(structures) == Atoms:
            structures = [structures]

        if self.sparseX is None:
            raise RuntimeError("Sparsex not defined for all descs. Use desc.set_sparsex().")

        N = len(structures)
        M = self.nsparse

        K = np.zeros((N, M))
        for i, structure in enumerate(structures):
            quip_result = self.quip_desc.calc(structure)

            if "covariance_cutoff" in quip_result.keys():
                # Has data for given descriptor
                x = quip_result["data"]
                x_cut = quip_result["covariance_cutoff"]

                k = self.cov_func(x, self.sparseX, x_cut, self.sparse_cuts, self.cov_prop)
                K[i, :] = np.sum(k, axis=0)
        return K

class GAPXMLWrapper():
    '''
    Small wrapper class to store key GAP information

    Key Attributes:
    self.gap_label : Label of the GAP potential
    self.isolated_atom_energies : dict of E0s; E0 for species Z is self.isolated_atom_energies[Z]
    self.num_desc : Number of descriptors
    self.descriptors : list of DescXMLWrapper objects for GAP descriptors
    self.total_nsparse : Number of sparse points for the GAP model
    self.weights : Full weights for the GAP model
    self.mean_weights : Mean weights for the original GAP model (IE: self.weights are changed for committors, self.mean_weights is fixed)

    Key Methods:
    self.save(fname) : Save the xml data back to an xml file
    self.as_potential() : Return the equivalent quippy.potential.Potential
    self.draw_posterior_sample(num_samples) : Draw samples from the posterior, if available
    '''

    def __init__(self, xml, mean_weights=None, xml_dir=None):
        self._xml_tree = xml
        root = self._xml_tree.getroot()


        if xml_dir is not None:
            self.xml_dir = xml_dir
        else:
            self.xml_dir = os.getcwd()

        self.gap_label = root[0].attrib["label"]

        iso_atom_energies = root[1][0][:]

        self.isolated_atom_energies = {}

        for item in iso_atom_energies:
            self.isolated_atom_energies[int(item.attrib["Z"])] = float(
                item.attrib["value"])

        descriptors = root[1][1][:]

        self.num_desc = len(descriptors)

        self.descriptors = [DescXMLWrapper(desc_xml)
                            for desc_xml in descriptors]

        self.weights = np.block([desc.weights for desc in self.descriptors])

        self.total_nsparse = self.weights.shape[0]

        if mean_weights is not None:
            self.mean_weights = mean_weights
        else:
            self.mean_weights = self.weights.copy()

        self.sparsex_loaded = False

    def load_sparsex(self):
        if not self.sparsex_loaded:
            for i, desc in enumerate(self.descriptors):
                sparse_path = self.path_to_xml + ".sparseX." + self.gap_label + str(i+1)
                sparseX = np.loadtxt(sparse_path).reshape((desc.nsparse, -1))
                desc.set_sparsex(sparseX)
            self.sparsex_loaded = True

    def save(self, fname):
        '''
        Save internal XML tree to fname
        '''

        file_dir = os.path.dirname(fname)

        if len(file_dir) == 0: # File should be saved to cwd
            dest_dir = os.getcwd()

        else:
            dest_dir = file_dir

        # Save XML file
        with open(fname, "wb") as f:
            self._xml_tree.write(f)

        # Find all sparseX files from source of GAP xml
        xml_src_files = [file for file in os.listdir(self.xml_dir) if ".sparseX."+self.gap_label in file]

        xml_dest_files = os.listdir(dest_dir)

        # Generate symlinks to original sparseX files
        for sparse_file in xml_src_files:
            if sparse_file not in xml_dest_files: # File does not exist in dir where xml will be saved
                os.symlink(self.xml_dir + os.sep + sparse_file, dest_dir + os.sep + sparse_file)

    def as_potential(self):
        '''
        Return quippy.potential.Potential equivalent to the model defined by the internal XML tree
        '''

        cwd = os.getcwd() # Remember which dir we're supposed to be in

        try:
            os.chdir(self.xml_dir) # Change to same dir as the xml, so QUIP can find sparseX files
            
            pot = Potential(param_str=tostring(self._xml_tree.getroot()))
            pot.xml = self

        except (TypeError, RuntimeError):
            raise FileNotFoundError(f"Conversion to quippy Potential failed, likely because sparseX files were not found in directory {self.xml_dir}")

        finally:
            os.chdir(cwd) # Ensure we always end in the same dir we started

        return pot

    def _posterior_sample(self):
        if self.R is not None:
            z = np.random.normal(size=self.total_nsparse)

            return self.mean_weights + np.linalg.solve(self.R, z)
        else:
            raise FileNotFoundError(f"R matrix not found in directory {self.xml_dir}.")

    def _xml_sample(self):
        new_weights = self._posterior_sample()

        new_root = fromstring(tostring(self._xml_tree.getroot()))

        new_tree = ElementTree(new_root)

        new_descs = new_root[1][1][:]

        isparse = 0

        for i in range(self.num_desc):

            desc_nsparse = self.descriptors[i].nsparse

            desc_wts = [child for child in new_descs[i]
                        if "sparseX" in child.tag]

            for j in range(desc_nsparse):
                desc_wts[j].attrib["alpha"] = str(new_weights[isparse])
                isparse += 1

        XYZ_data = [child for child in new_root[1] if child.tag == "XYZ_data"]

        if len(XYZ_data):  # Parent xml has XYZ data
            for dat in XYZ_data:
                new_root[1].remove(dat)

        return new_tree

    def draw_posterior_samples(self, num_samples=1):
        '''
        Draw samples from the posterior of the GAP model
        Only possible if <GAP_fname>.R.<GAP_label> exists in the same dir as the GAP XML file
        '''
        if num_samples == 1:
            return GAPXMLWrapper(self._xml_sample(), mean_weights=self.mean_weights, xml_dir=self.xml_dir)
        else:
            return [self.draw_posterior_samples() for i in range(num_samples)]


def read_xml(path_to_xml):
    '''
    Generate an instance of GAPXMLWrapper for given XML file
    '''

    xml_dir = os.path.dirname(path_to_xml)

    if len(xml_dir) == 0: # XML in cwd
        xml_dir = os.getcwd()

    
    gap = GAPXMLWrapper(parse(path_to_xml), xml_dir=xml_dir)
    gap.path_to_xml = path_to_xml

    R_fname = path_to_xml + ".R." + gap.gap_label

    if os.path.exists(R_fname):
        gap.R = np.loadtxt(R_fname).reshape(
            (gap.total_nsparse, gap.total_nsparse)).T
    else:
        gap.R = None

    return gap


def get_xml_committee(path_to_xml, committee_size, return_core_wrapper=False):
    '''
    Sample a committee of GAPXMLWrappers based on the model defined by path_to_xml XML file
    '''
    gap_wrapper = read_xml(path_to_xml)

    committee = gap_wrapper.draw_posterior_samples(committee_size)

    if return_core_wrapper:
        return committee, gap_wrapper
    else:
        return committee


def get_calc_committee(path_to_xml, committee_size, return_core_wrapper=False):
    '''
    Sample a committee of quippy.potential.Potentials based on the model defined by path_to_xml XML file
    '''
    xml_committee, gap_wrapper = get_xml_committee(
        path_to_xml, committee_size, return_core_wrapper=True)
    calc_committee = [model.as_potential() for model in xml_committee]

    if return_core_wrapper:
        return calc_committee, gap_wrapper
    else:
        return calc_committee


def gap_score(gapxml, structures, sparsifier=get_cur_scores, existing_dataset=[], descriptor_weights=1.0, existing_dataset_cache_name=None):
    '''
    Compute "GAP Scores" for sampling a sparse set of the input structures

    Uses the sparse GP defined by gapxml to compute the design matrix associated with both structures and existing_dataset,
        and use the provided sparsifier to score entries of the design matrix.

    2B descriptors are especially slow to compute at the design matrix level, and don't contribute much information to the system.
    They can be essentially "turned off" by setting their descriptor_weights to zero (E.G. descriptor_weights=np.array([0.0, 0.0, 0.0, 1.0, 1.0]))
        for a SOAP-only scoring of a 2-species 2B+SOAP GAP

    gapxml: GAPXMLWrapper object
        GAP model to use in computing the design matrix
    structures: list of ase Atoms objects
        Structures to compute scores for
    sparsifier: function f(vecs) -> scores
        Function which computes N scores given an NxM matrix of vector quantities
    existing_dataset: list of ase Atoms objects
        "Core" structures which are not scored, but influence the scoring of structures
        The vectors of both structures and existing_dataset are passed to sparsifier, but only scores
        associated with structures are used
    descriptor_weights: array of floats
        Descriptor-level weightings to bias scoring towards a particular descriptor.
    existing_dataset_cache_name: string
        Filename passed to np.save, to cache the design matrix entries for existing_dataset (to save on recomputation)

    returns an array of scores, len(scores) == len(structures), np.sum(scores) == 1
    '''

    N_existing = len(existing_dataset)
    N_struct = len(structures)
    N = N_existing + N_struct

    Ms = [desc.nsparse for desc in gapxml.descriptors]
    M = sum(Ms)

    ndesc = gapxml.num_desc

    if np.issubdtype(type(descriptor_weights), np.floating) or np.issubdtype(type(descriptor_weights), np.integer):
        weights = np.ones(ndesc)
    else:
        assert len(descriptor_weights) == ndesc
        weights = np.abs(np.array(descriptor_weights))

        # Only relative scaling matters, ensure numbers are "nice"
        weights /= np.max(weights)

    # Ensure sparsex have been read from file
    gapxml.load_sparsex()

    have_existing_design = False
    if existing_dataset_cache_name is not None and N_existing > 0:
        try:
            K_existing = np.load(existing_dataset_cache_name + ".npy")
            # Make sure design matrix is of correct shape
            assert K_existing.shape == (N_existing, M)
        except (FileNotFoundError, AssertionError) as e:
            # Fine if no K found, just needs to be recomputed
            if type(e) == AssertionError:
                print(f"existing_dataset design matrix was of the wrong shape (got {K_existing.shape}, expected {(N_existing, M)}). Recomputing...")
            pass
        else:
            # else clause is triggered iff no exceptions were raised
            have_existing_design = True

    if not have_existing_design and N_existing > 0:
        # Compute design matrix for existing_dataset
        K_existing = np.zeros((N_existing, M))

        m_start = 0

        for i, desc in enumerate(gapxml.descriptors):
            if weights[i] < 1e-3: # Weight too small, ignore descriptor
                continue
            
            K_existing[:, m_start:m_start + desc.nsparse] = desc.get_energy_design(existing_dataset)
            m_start += desc.nsparse
        
        # Cache K_existing for reruns
        if existing_dataset_cache_name is not None: 
            np.save(existing_dataset_cache_name, K_existing)

    # Compute the design matrix for the input structures
    K_struct = np.zeros((N_struct, M))

    m_start = 0

    for i, desc in enumerate(gapxml.descriptors):
        if weights[i] < 1e-3: # Weight too small, ignore descriptor
            continue
        
        K_struct[:, m_start:m_start + desc.nsparse] = desc.get_energy_design(structures)
        m_start += desc.nsparse

    if N_existing > 0:
        K = np.concatenate((K_existing, K_struct), axis=0)
    else:
        K = K_struct


    scores = sparsifier(K)[N_existing:]

    print(N_struct, scores.shape)

    scores /= np.sum(scores)

    return scores