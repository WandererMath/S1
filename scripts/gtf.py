import os
from dataclasses import dataclass
from warnings import warn
import random
from utils import *


import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from intervaltree import IntervalTree, Interval
import gffutils
import gffutils.attributes
import gffutils.feature
from hyper import TIR_DOWNSTREAM, TIR_UPSTREAM


# START_CODONS=['ATG', 'GTG', 'TTG'] # AUG is the only one that codes for Methionine
START_CODONS=['ATG']
STOP_CODONS=['TAA', 'TAG', 'TGA']


def seq_gc_content(seq):
    seq=str(seq)
    c=seq.count('G')+seq.count('C')
    total=len(seq)
    return c/total



class GTF:
    '''
    CDS as key
    Needs to write a new class if rRNA seqs are needed.
    '''
    db: gffutils.FeatureDB
    fna: Seq
    fna_description: dict
    def __init__(self, gtf_file, fna_file=None, id_spec=None):
        is_gff3 = gtf_file.lower().endswith((".gff", ".gff3"))
        # Version the cache name: prior caches used GTF's gene_id convention
        # and cannot identify NCBI GFF3 CDS records.
        db_path = f"{gtf_file}.gffutils-v2.db"
        if id_spec is None:
            id_spec = {"CDS": "ID"} if is_gff3 else {"CDS": "gene_id"}
        if os.path.exists(db_path):
            self.db = gffutils.FeatureDB(db_path, keep_order=True)
        else:
            self.db = gffutils.create_db(gtf_file, dbfn=db_path, force=True, keep_order=True, merge_strategy='merge', \
                                         id_spec=id_spec)
    
        if fna_file is not None:
            self.fna={}
            self.fna_description={}
            with open(fna_file, 'r') as fna_file:
                for record in SeqIO.parse(fna_file, 'fasta'):
                    self.fna[record.name]=Seq(record.seq)
                    self.fna_description[record.name]=record.description

    def all_genes(self):
        """Return database IDs for every annotated CDS feature.

        NCBI GFF3 records use ``ID=cds-...`` rather than GTF's ``gene_id``.
        """
        return [feature.id for feature in self.db.features_of_type('CDS')]


    def id2name(self, gene_id):
        gene_feature = self.db[gene_id]
        return gene_feature.attributes.get('gene', [None])[0] 
    
    def get_info_from_id(self, gene_id):
        '''
            Return:
                A dict
        '''
        gene_feature = self.db[gene_id] 
        gene_feature: gffutils.feature.Feature
        #breakpoint()
        return gene_feature.attributes.__dict__['_d']
    
    def save_IDs_info(self, IDs, output, fields=['gene_id', 'gene', 'gene_biotype', 'product', 'go_function', 'go_process']):        
        # df=[[self.db[g].attributes.get(f, [''])[0] for f in fields] for g in IDs]
        df=[]
        for g in IDs:
            try: 
                df.append([self.db[g].attributes.get(f, [''])[0] for f in fields])
            except:
                pass
        df=pd.DataFrame(df, columns=fields)
        df.to_csv(output, index=False, header=True)


    def id2protein(self, gene_id, field='product'):
        info_dict=self.get_info_from_id(gene_id)
        if field in info_dict:
            return info_dict[field][0]
        else:
            return None

    def name2id(self, name):
        for id in self.all_genes():
            if self.id2name(id)==name:
                return id
            
    
    
    ### Chrom Considered
    def get_seq_from_to(self, start, end, strand, chrom):
        """
        Get the sequence from start to end.
        Arguments are python 0-based slice index
        Args:
            start (int): Start position.
            end (int): End position.
            strand (str): Strand, either '+' or '-'.
        Returns:
            str: The sequence from start to end.
            Already reversed complemented if strand is '-'
        """
        if strand == '+':
            return str(self.fna[chrom][start:end])
        elif strand == '-':
            seq = self.fna[chrom][start:end]
            return str(seq.reverse_complement())
    
    def get_gene_seq(self, gene_id, protein_coding=False):
        try:
            gene = self.db[gene_id]
            #breakpoint()
            # 1-based index
            result=self.get_seq_from_to(gene.start - 1, gene.end, gene.strand, gene.chrom)
            if protein_coding:
                assert len(result)%3==0
            return result
        except KeyError:
            print(f"Gene ID {gene_id} not found in the GTF file.")
    
    def get_gene_positions_strand_chrom(self, gene_id):
        '''
            Return positions are 0-based index [) ready to use 
            Return:
                start, end, strand, chrom
        '''
        try:
            gene = self.db[gene_id]
            return gene.start - 1, gene.end, gene.strand, gene.chrom
        except KeyError:
            print(f"Gene ID {gene_id} not found in the GTF file.")
    
    def get_seq_by_gene_and_offset(self, gene_id, offset1, offset2):
        '''
        Length= offset2 - offset1
        1= AUG's A
        '''
        assert offset2 > offset1, "offset2 must be greater than offset1"
        start, end, strand, chrom = self.get_gene_positions_strand_chrom(gene_id)
        if strand == '+':
            return self.get_seq_from_to(start + offset1, start + offset2, strand, chrom)
        elif strand == '-':
            return self.get_seq_from_to(end - offset2, end - offset1, strand, chrom)
        else:
            raise Exception(f"Strand {strand} not recognized. Only '+' and '-' are allowed.")
    ##############


    ## ML Data Preparation
    
        
    def all_start_positions(self, chrom):
        '''
            Return:
                {'+':[int, ...], '-':[]}
        '''
        results = {'+': [], '-': []}

        for gene_id in self.all_genes():
            try:
                gene_record=self.db[gene_id]
            except KeyError:
                print(f"Gene ID {gene_id} not found in the GTF file.")
                breakpoint()
                continue
            if gene_record.chrom == chrom:
                results[gene_record.strand].append(gene_record.start-1 if gene_record.strand=='+' else gene_record.end)

        return results
    
    def all_CDS_intervals(self, chrom):
        '''
            Return:
                {'+':[int, ...], '-':[]}
        '''
        results = {'+': IntervalTree(), '-': IntervalTree()}
        for gene_id in self.all_genes():
            gene_record=self.db[gene_id]
            start, end, strand=gene_record.start, gene_record.end, gene_record.strand

            if self.db[gene_id].chrom == chrom:
                results[strand].add(Interval(start-3, end+3))
                
        return results
    
    def get_all_candidate_start_positions(self, chrom, strand):
        fna=self.fna[chrom]
        if strand=='+':
            return [i for i in range(len(fna) - 2)
            if fna[i:i+3] in START_CODONS]
        elif strand=='-':
            START_CODONS_REVERSE_COMP = [Seq(codon).reverse_complement() for codon in START_CODONS]
            return [i+3 for i in range(len(fna) - 2)
            if fna[i:i+3] in START_CODONS_REVERSE_COMP]
        
        else:
            raise ValueError(f"Strand {strand} not recognized. Only '+' and '-' are allowed.")

            
    def __get_all_candidate_start_positions_in_CDS(self, chrom, strand):
        fna=self.fna[chrom]
        tree=self.all_CDS_intervals(chrom)[strand]
        #breakpoint()
        if strand=='+':
            return [i for i in range(len(fna) - 2)
            if fna[i:i+3] in START_CODONS and match2start(tree,i) is not None]
        
        elif strand=='-':
            START_CODONS_REVERSE_COMPLEMENT = [Seq(codon).reverse_complement_rna() for codon in START_CODONS]
            return [i+3 for i in range(len(fna) - 2)
            if fna[i:i+3] in START_CODONS_REVERSE_COMPLEMENT and match2start(tree,i) is not None]
        
        else:
            raise ValueError(f"Strand {strand} not recognized. Only '+' and '-' are allowed.")


    def get_tir_by_position(self, position,chrom, strand,
                            l=-TIR_UPSTREAM, h=3 + TIR_DOWNSTREAM):
        """
        Get the TIR sequence around a given position.
        Args:
            position (int): The position in the sequence.
            l (int): Length to extend to the left.
            h (int): Length to extend to the right.
        Returns:
            str: The TIR sequence.
            None: if not enough sequence available.
        """
        fna=self.fna[chrom]
        if strand == '+':
            start = max(0, position + l)
            end = min(len(fna), position + h)
            if end-start!=h-l:
                return None
            return str(fna[start:end])
        elif strand == '-':
            start = max(0, position - h)
            end = min(len(fna), position - l)
            if end-start!=h-l:
                return None
            seq = fna[start:end]
            if any(x in seq for x in ['Y', 'N', 'R']):
                return None
            return str(seq.reverse_complement())
    def gene_to_tir(self, gene_id, l=-TIR_UPSTREAM, h=3 + TIR_DOWNSTREAM):
        """
        Get the TIR sequence for a gene.
        Args:
            gene_id (str): The ID of the gene.
            l (int): Length to extend to the left.
            h (int): Length to extend to the right.
        Returns:
            str: The TIR sequence.
        """
        start, end, strand, chrom = self.get_gene_positions_strand_chrom(gene_id)
        return self.get_tir_by_position(start if strand=='+' else end, chrom, strand, l, h)
    
    def get_CDS(self, gene_id, a=25, b=25):
        start, end, strand, chrom= self.get_gene_positions_strand_chrom(gene_id)
        if strand == '+':
            return self.get_seq_from_to(start - a, end + b, strand, chrom)
        elif strand == '-':
            return self.get_seq_from_to(start - b, end + a, strand, chrom)
        else:
            raise Exception(f"Strand {strand} not recognized. Only '+' and '-' are allowed.")



    def prepare_balanced_data(self):
        """
        Return:
            [seqs], [labels]
            label: 1 Gene, 0 Not Gene
        """
        seqs=[]
        labels=[]
        for chrom in self.fna:
            tmp=self.all_start_positions(chrom)
            for strand in ['+', '-']:
                position= self.get_all_candidate_start_positions(chrom, strand)
                gene_position=tmp[strand]
                #breakpoint()
                # seqs+=[self.get_tir_by_position(p, chrom,  strand) for p in position if self.get_tir_by_position(p, chrom, strand) is not None]
                # labels+=[1 if p in gene_position else 0 for p in position if self.get_tir_by_position(p, chrom, strand) is not None]
                tmp_seqs=[]
                tmp_labels=[]
                for p in position:
                    if (self.get_seq_from_to(p, p+3, strand, chrom)!='ATG' and strand=='+')\
                        or (self.get_seq_from_to(p-3, p, strand, chrom)!='ATG' and strand=='-'):
                            continue
                    seq=self.get_tir_by_position(p, chrom, strand)

                    if seq is not None:
                        label=1 if p in gene_position else 0
                        # Balance 
                        if label==1 or random.random()<0.0224: 
                            tmp_seqs.append(seq)
                            tmp_labels.append(label)
                seqs+=tmp_seqs
                labels+=tmp_labels
        assert len(labels)==len(seqs), f"Labels and sequences length mismatch: {len(labels)} != {len(seqs)}"
        return seqs, labels

    def prepare_data(self):
        """
        Return:
            [seqs], [labels]
            label: 1 Gene, 0 Not Gene
        """
        seqs=[]
        labels=[]
        for chrom in self.fna:
            tmp=self.all_start_positions(chrom)
            for strand in ['+', '-']:
                position= self.get_all_candidate_start_positions(chrom, strand)
                gene_position=tmp[strand]
                #breakpoint()
                # seqs+=[self.get_tir_by_position(p, chrom,  strand) for p in position if self.get_tir_by_position(p, chrom, strand) is not None]
                # labels+=[1 if p in gene_position else 0 for p in position if self.get_tir_by_position(p, chrom, strand) is not None]
                tmp_seqs=[]
                tmp_labels=[]
                for p in position:
                    if (self.get_seq_from_to(p, p+3, strand, chrom)!='ATG' and strand=='+')\
                        or (self.get_seq_from_to(p-3, p, strand, chrom)!='ATG' and strand=='-'):
                            continue
                    seq=self.get_tir_by_position(p, chrom, strand)

                    if seq is not None:
                        label=1 if p in gene_position else 0
                        tmp_seqs.append(seq)
                        tmp_labels.append(label)
                seqs+=tmp_seqs
                labels+=tmp_labels
        assert len(labels)==len(seqs), f"Labels and sequences length mismatch: {len(labels)} != {len(seqs)}"
        return seqs, labels
    
    def prepare_data_encoded(self):
        seqs, labels=self.prepare_data()
        seqs_encoded=[]
        filtered_labels=[]
        for seq, label in zip(seqs, labels):
            if 'N' in str(seq).upper():
                continue
            try:
                encoded = rna_encoding(seq)
            except KeyError:
                continue
            seqs_encoded.append(encoded)
            filtered_labels.append(label)
        return seqs_encoded, filtered_labels




    def prep_data_in_CDS_encoded(self):
        """
        Return:
            [seqs], [labels]
            label: 1 Gene, 0 Not Gene
        """
        seqs=[]
        labels=[]
        for chrom in self.fna:
            tmp=self.all_start_positions(chrom)
            for strand in ['+', '-']:
                position= self.get_all_candidate_start_positions_in_CDS(chrom, strand)
                gene_position=tmp[strand]
                # seqs+=[self.get_tir_by_position(p, chrom,  strand) for p in position if self.get_tir_by_position(p, chrom, strand) is not None]
                # labels+=[1 if p in gene_position else 0 for p in position if self.get_tir_by_position(p, chrom, strand) is not None]
                tmp_seqs=[]
                tmp_labels=[]
                for p in position:
                    if (self.get_seq_from_to(p, p+3, strand, chrom)!='ATG' and strand=='+')\
                        or (self.get_seq_from_to(p-3, p, strand, chrom)!='ATG' and strand=='-'):
                            continue
                    seq=self.get_tir_by_position(p, chrom, strand)
                    if seq is not None:
                        label=1 if p in gene_position else 0
                        # Balance 
                        if label==1 or random()<0.0224*2.41: # 10% of non-gene sequences
                            tmp_seqs.append(seq)
                            tmp_labels.append(label)
                seqs+=tmp_seqs
                labels+=tmp_labels
        assert len(labels)==len(seqs), f"Labels and sequences length mismatch: {len(labels)} != {len(seqs)}"
        from mylib.utils import rna_encoding
        seqs_encoded=[rna_encoding(seq) for seq in seqs]
        return seqs_encoded, labels
    
    

    def test(self):
        protein_coding=0
        non_protein_coding=0
        only_start_codon_wrong=0
        only_stop_codon_wrong=0
        both_wrong=0
        correct=0
        for g in self.all_genes():
            try:
                seq=self.get_gene_seq(g, protein_coding=True)
                protein_coding+=1
                start_correct= seq[:3] in START_CODONS 
                stop_correct=seq[-3:] in STOP_CODONS
                if start_correct and stop_correct:
                    correct+=1
                elif start_correct and not stop_correct:
                    only_stop_codon_wrong+=1
                elif not start_correct and stop_correct:
                    only_start_codon_wrong+=1
                else:
                    both_wrong+=1

            except AssertionError:
                non_protein_coding+=1

        print(f"Potential Protein coding genes: {protein_coding}")
        print(f"Non-protein coding genes: {non_protein_coding}")
        print(f"Correct start and stop codon: {correct}")
        print(f"Only start codon wrong: {only_start_codon_wrong}")
        print(f"Only stop codon wrong: {only_stop_codon_wrong}")
        print(f"Both start and stop codon wrong: {both_wrong}")
    
    def id2length(self, gene_id):
        a, b, _, _=self.get_gene_positions_strand_chrom(gene_id)
        return b-a    
            
    def is_protein(self, gene_id): 
        warn("Deprecated") 
        feature=self.db[gene_id]    
        biotype=feature.attributes.get('gene_biotype', [''])[0]
        if biotype=='protein_coding':
            return True
        return False
    
    def id2biotype(self, gene_id):
        feature=self.db[gene_id]
        return feature.attributes.get('gene_biotype', [''])[0]
    def gene_chrom_summary(self):
        genes= self.all_genes()
        summary = {}
        for gene_id in genes:
            feature = self.db[gene_id]
            chrom = feature.chrom
            if chrom not in summary:
                summary[chrom] = 0
            summary[chrom]+=1
        for chrom in summary:
            print( f"{chrom}: {summary[chrom]} ")
        return summary


    def get_start_and_direction(self, gene_id):

        # !!!
        ## GTF File:
        ## Start, End positions are 1-based index. Both ends included.
        try:
            gene = self.db[gene_id]
            strand=self.db[gene_id].strand
            #breakpoint()
            if strand=="+":
                return gene.start, strand
            else:
                return gene.end, strand
            #print(f"The start position of gene {gene_id} is {start_position}")
        except KeyError:
            print(f"Gene ID {gene_id} not found in the GTF file.")

    #####################################################
    ## Deprecated methods
    #####################################################

    
    def get_low_high_direction(self, gene_id):
        warn("Deprecated", category=UserWarning)
        try:
            gene = self.db[gene_id]
            return gene.start, gene.end, gene.strand
        except KeyError:
            print(f"Gene ID {gene_id} not found in the GTF file.")







@dataclass
class Aligner:
    ASD: str
    pairing_rule: list # list of sets
    def _nt_pairs(self, a, b)->bool:
        if any(set([a, b])==rule for rule in self.pairing_rule):
            return True
        return False
    def _pair1(self, seq, n):
        # align at n-th of the long 
        # return staring index (of the long) of the longest pairing region and the pairing length
        tmp={}
        j=n
        for i in range(n, len(self.ASD)+n):
            if self._nt_pairs(seq[i],self.ASD[i-n]):
                if j in tmp:
                    tmp[j]+=1
                else:
                    tmp[j]=1
            else:
                j=i+1
        max_key = max(tmp, key=tmp.get)
        max_value = tmp[max_key]
        return max_key, max_value
    
    def pair(self, seq):
        tmp=[]
        for i in range(len(seq)-len(self.ASD)+1):
            r=self._pair1(seq, i)
            tmp.append(r)
        return max(tmp, key=lambda x: x[1])
    def output_pairing(self, seq):
        i, l=self.pair(seq)
        return f"{seq[:i]}[{seq[i:i+l]}]{seq[i+l:]}", i, l


if __name__=='__main__':
    FILE_FNA='GCF_000007825.1_ASM782v1_genomic.fna'
    FILE_GTF='genomic.gtf'
    gtf=GTF(FILE_GTF, FILE_FNA)
    gene1=gtf.all_genes()[0]
    gtf.is_protein(gene1)
    print(gtf.id2length(gene1))

    
