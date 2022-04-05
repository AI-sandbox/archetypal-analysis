"""
Created on Monday April 4 2022
@author: Albert Dominguez
This function has been modified from https://github.com/ai-sandbox/neural-admixture/src/snp_reader.py
"""
import numpy as np
import sys

class SNPReader:
    def _read_vcf(self, file):
        print('Input format is VCF.')
        import allel
        f_tr = allel.read_vcf(file)
        G = f_tr['calldata/GT']
        G[G<0] = 0
        return np.sum(G, axis=2).T/2
    
    def _read_bed(self, file):
        print('Input format is BED.')
        from pandas_plink import read_plink
        _, _, G = read_plink('.'.join(file.split('.')[:-1]))
        G_loaded = (G.T/2).compute()
        G_loaded[np.isnan(G_loaded)] = 0
        return G_loaded
    
    def _read_pgen(self, file):
        print('Input format is PGEN.')
        try:
            import pgenlib as pg
        except ImportError as ie:
            print('ERROR: Cannot read PGEN file as pgenlib is not installed.')
            sys.exit(1)
        except Exception as e:
            raise e
        file_prefix = file.split('.pgen')[0]
        pgen, _, _ = str.encode(file), f'{file_prefix}.psam', f'{file_prefix}.pvar' # Genotype, sample, variant files
        pgen_reader = pg.PgenReader(pgen)
        calldata = np.ascontiguousarray(np.empty((pgen_reader.get_variant_ct(), 2*pgen_reader.get_raw_sample_ct())).astype(np.int32))
        pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
        return (calldata[:,::2]+calldata[:,1::2]).T/2
    
    def read_data(self, file):
        if file.endswith('.vcf') or file.endswith('.vcf.gz'):
            G = self._read_vcf(file)
        elif file.endswith('.bed'):
            G = self._read_bed(file)
        elif file.endswith('.pgen'):
            G = self._read_pgen(file)
        else:
            print('ERROR: Invalid format. Unrecognized file format. Make sure file ends with .vcf | .vcf.gz | .bed | .pgen')
            sys.exit(1)
        assert int(G.min()) == 0 and int(G.max()) == 1, 'Only biallelic SNPs are supported. Please make sure multiallelic sites have been removed and missing values have been imputed.'
        return G if np.mean(G) < 0.5 else 1-G