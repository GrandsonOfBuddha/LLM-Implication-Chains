import json
import jsonlines
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import openai
import re
import time
import networkx as nx
from itertools import combinations


@dataclass
class Atom:
    """Lightweight symbolic representation of a statement"""
    subject: str
    predicate: str
    object: str
    quantifier: str = "some"  # all, some, no, most
    polarity: bool = True  # True for positive, False for negative
    
    def __str__(self):
        polarity_str = "" if self.polarity else "not "
        return f"{self.quantifier} {self.subject} {polarity_str}{self.predicate} {self.object}"


@dataclass 
class Statement:
    """A single statement with text and symbolic representation"""
    text: str
    atoms: List[Atom]
    id: Optional[str] = None


class ImplicationChainGenerator:
    """Generates implication chains using ChatGPT API"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def extract_atoms(self, statement_text: str) -> List[Atom]:
        """Extract symbolic atoms from statement text using simple rules"""
        atoms = []
        
        # Simple pattern matching for basic logical forms
        patterns = [
            (r"All (\w+) are (\w+)", lambda m: Atom(m.group(1), "are", m.group(2), "all", True)),
            (r"No (\w+) are (\w+)", lambda m: Atom(m.group(1), "are", m.group(2), "no", True)),
            (r"Some (\w+) are (\w+)", lambda m: Atom(m.group(1), "are", m.group(2), "some", True)),
            (r"(\w+) can (\w+)", lambda m: Atom(m.group(1), "can", m.group(2), "some", True)),
            (r"(\w+) cannot (\w+)", lambda m: Atom(m.group(1), "can", m.group(2), "some", False)),
        ]
        
        for pattern, atom_func in patterns:
            matches = re.finditer(pattern, statement_text, re.IGNORECASE)
            for match in matches:
                atoms.append(atom_func(match))
        
        # If no patterns match, create a generic atom
        if not atoms:
            words = statement_text.split()
            if len(words) >= 3:
                atoms.append(Atom(words[0], words[1], words[2], "some", True))
        
        return atoms
    
    def generate_implication_with_context(self, conversation_history: List[Dict]) -> Tuple[str, List[Atom]]:
        """Generate a single logical implication with full conversation context"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                max_tokens=100,
                temperature=0.3
            )
            
            implication_text = response.choices[0].message.content.strip()
            atoms = self.extract_atoms(implication_text)
            
            return implication_text, atoms
            
        except Exception as e:
            print(f"Error generating implication: {e}")
            # Return the last statement as fallback
            last_statement = conversation_history[-1]["content"].split('"')[1] if '"' in conversation_history[-1]["content"] else "Default statement"
            return last_statement, self.extract_atoms(last_statement)
    
    def generate_implication(self, premise: str) -> Tuple[str, List[Atom]]:
        """Generate a single logical implication from a premise (legacy method)"""
        conversation = [
            {"role": "system", "content": "You are a logical reasoning expert. Generate clear, factual implications from premises."},
            {"role": "user", "content": f'Given the premise: "{premise}"\n\nGenerate ONE statement that logically follows from this premise. The statement should be:\n1. A clear logical consequence\n2. More specific or contain additional information\n3. Factually consistent with the premise\n\nRespond with only the implied statement, nothing else.'}
        ]
        
        return self.generate_implication_with_context(conversation)
    
    def generate_chain(self, seed_statement: str, chain_length: int = 3) -> List[Statement]:
        """Generate a complete implication chain from a seed statement with conversation context"""
        chain = []
        
        # Initialize conversation with system context
        conversation_history = [
            {
                "role": "system", 
                "content": "You are a logical reasoning expert. Your task is to build a coherent chain of logical implications. Each new statement should:\n1. Logically follow from the previous statements\n2. Be more specific or add new factual information\n3. Maintain consistency with all previous statements\n4. Be factually accurate"
            }
        ]
        
        current_text = seed_statement
        
        for i in range(chain_length + 1):  # +1 to include the seed
            atoms = self.extract_atoms(current_text)
            statement = Statement(text=current_text, atoms=atoms, id=f"stmt_{i}")
            chain.append(statement)
            
            # Add current statement to conversation history
            if i == 0:
                conversation_history.append({
                    "role": "user",
                    "content": f'Starting with the premise: "{current_text}"\n\nGenerate the next logical statement that follows from this premise.'
                })
            else:
                conversation_history.append({
                    "role": "assistant",
                    "content": current_text
                })
            
            if i < chain_length:  # Don't generate after the last statement
                # Build context-aware prompt for next implication
                chain_context = " → ".join([stmt.text for stmt in chain])
                
                conversation_history.append({
                    "role": "user",
                    "content": f'Current chain: {chain_context}\n\nGenerate the next logical statement that follows from "{current_text}". Ensure it:\n1. Logically follows from the current statement\n2. Is consistent with the entire chain\n3. Adds specific factual information\n\nRespond with only the next statement, nothing else.'
                })
                
                time.sleep(0.5)  # Rate limiting
                current_text, _ = self.generate_implication_with_context(conversation_history)
        
        return chain
    
    def generate_chain_batch(self, seed_statements: List[str], chain_length: int = 3) -> List[List[Statement]]:
        """Generate multiple independent chains in batch for efficiency"""
        chains = []
        
        print(f"Generating {len(seed_statements)} independent chains...")
        
        for i, seed in enumerate(seed_statements):
            print(f"\n--- Chain {i+1}/{len(seed_statements)} ---")
            print(f"Seed: {seed}")
            
            try:
                # Each chain is completely independent
                chain = self.generate_chain(seed, chain_length)
                chains.append(chain)
                
                print("Generated chain:")
                for j, stmt in enumerate(chain):
                    print(f"  {j}: {stmt.text}")
                
            except Exception as e:
                print(f"Error generating chain from '{seed}': {e}")
                # Create minimal chain with just the seed
                atoms = self.extract_atoms(seed)
                fallback_chain = [Statement(text=seed, atoms=atoms, id="stmt_0")]
                chains.append(fallback_chain)
        
        return chains
    
    def validate_chain_coherence(self, chain: List[Statement]) -> Dict[str, Any]:
        """Validate that a chain maintains logical coherence"""
        validation_results = {
            "is_coherent": True,
            "issues": [],
            "atom_consistency": True,
            "length_valid": len(chain) > 1
        }
        
        # Check for atom contradictions within the chain
        all_atoms = []
        for stmt in chain:
            all_atoms.extend(stmt.atoms)
        
        # Look for contradictory atoms
        for i, atom1 in enumerate(all_atoms):
            for atom2 in all_atoms[i+1:]:
                if (atom1.subject == atom2.subject and 
                    atom1.predicate == atom2.predicate and 
                    atom1.object == atom2.object):
                    
                    if ((atom1.quantifier == "all" and atom2.quantifier == "no") or
                        (atom1.quantifier == "no" and atom2.quantifier == "all") or
                        (atom1.polarity != atom2.polarity)):
                        
                        validation_results["is_coherent"] = False
                        validation_results["atom_consistency"] = False
                        validation_results["issues"].append(f"Contradictory atoms: {atom1} vs {atom2}")
        
        return validation_results


class ChainGraphBuilder:
    """Builds directed graphs from implication chains"""
    
    def build_graph(self, chains: List[List[Statement]]) -> nx.DiGraph:
        """Build a directed graph from multiple implication chains"""
        G = nx.DiGraph()
        
        for chain_idx, chain in enumerate(chains):
            for i, statement in enumerate(chain):
                node_id = f"chain_{chain_idx}_stmt_{i}"
                G.add_node(node_id, statement=statement)
                
                # Add edge to next statement in chain
                if i < len(chain) - 1:
                    next_node_id = f"chain_{chain_idx}_stmt_{i+1}"
                    G.add_edge(node_id, next_node_id, relation="entails")
        
        return G


class PairSampler:
    """Samples and labels statement pairs from chains"""
    
    def __init__(self, min_distance: int = 2):
        self.min_distance = min_distance
    
    def atoms_contradict(self, atoms1: List[Atom], atoms2: List[Atom]) -> bool:
        """Check if two sets of atoms contradict each other"""
        for a1 in atoms1:
            for a2 in atoms2:
                if (a1.subject == a2.subject and 
                    a1.predicate == a2.predicate and 
                    a1.object == a2.object):
                    
                    # Check for quantifier contradictions
                    if ((a1.quantifier == "all" and a2.quantifier == "no") or
                        (a1.quantifier == "no" and a2.quantifier == "all") or
                        (a1.polarity != a2.polarity)):
                        return True
        
        return False
    
    def sample_pairs_from_chain(self, chain: List[Statement]) -> List[Dict]:
        """Sample pairs from a single chain with automatic labeling"""
        pairs = []
        
        for i in range(len(chain)):
            for j in range(i + self.min_distance, len(chain)):
                premise = chain[i]
                hypothesis = chain[j]
                
                # Label based on position and atom analysis
                if j > i:  # Forward direction in chain
                    label = "entails"
                elif self.atoms_contradict(premise.atoms, hypothesis.atoms):
                    label = "contradicts"
                else:
                    label = "independent"
                
                pairs.append({
                    "premise": premise.text,
                    "hypothesis": hypothesis.text,
                    "label": label,
                    "chain_distance": j - i
                })
        
        return pairs
    
    def sample_cross_chain_pairs(self, chains: List[List[Statement]], 
                                num_pairs: int = 1000) -> List[Dict]:
        """Sample pairs across different chains"""
        pairs = []
        all_statements = []
        
        # Flatten all chains
        for chain in chains:
            all_statements.extend(chain)
        
        # Sample random pairs
        import random
        for _ in range(min(num_pairs, len(all_statements) ** 2)):
            stmt1, stmt2 = random.sample(all_statements, 2)
            
            if self.atoms_contradict(stmt1.atoms, stmt2.atoms):
                label = "contradicts"
            else:
                label = "independent"
            
            pairs.append({
                "premise": stmt1.text,
                "hypothesis": stmt2.text,
                "label": label,
                "chain_distance": -1  # Cross-chain
            })
        
        return pairs


def create_seed_statements() -> List[str]:
    """Create a set of seed statements for the MVP"""
    return [
        "All dogs are mammals",
        "Birds can fly",
        "Water boils at 100 degrees Celsius",
        "The Earth orbits the Sun",
        "Humans need oxygen to survive",
        "All squares are rectangles",
        "Fish live in water",
        "Trees produce oxygen",
        "Cats are carnivores",
        "The sun rises in the east",
        "Ice melts when heated",
        "Books contain information",
        "Computers process data",
        "Rain falls from clouds",
        "Mountains are tall",
        "Fire produces heat",
        "Flowers bloom in spring",
        "Cars need fuel to run",
        "Stars emit light",
        "Salt dissolves in water",
        "Magnets attract metal",
        "Phones enable communication",
        "Plants need sunlight",
        "Snow is cold",
        "Doctors treat patients",
        "Teachers educate students",
        "Farmers grow crops",
        "Engineers design systems",
        "Artists create art",
        "Musicians play instruments",
        "Writers produce texts",
        "Chefs prepare food",
        "Pilots fly airplanes",
        "Drivers operate vehicles",
        "Scientists conduct research",
        "Athletes compete in sports",
        "Students attend classes",
        "Patients visit hospitals",
        "Customers buy products",
        "Workers earn wages",
        "Parents raise children",
        "Friends support each other",
        "Leaders make decisions",
        "Experts provide advice",
        "Tourists visit places",
        "Residents live in homes",
        "Citizens follow laws",
        "Voters elect representatives",
        "Consumers purchase goods",
        "Investors seek returns"
    ]


if __name__ == "__main__":
    # Example usage
    seeds = create_seed_statements()[:10]  # Use first 10 for testing
    print(f"Created {len(seeds)} seed statements")
    print("First few seeds:", seeds[:3])