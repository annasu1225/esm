from typing import Any
import torch

from esm.models.esm3 import ESM3
from esm.utils.structure.affine3d import build_affine3d_from_coordinates
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C



def get_backbone_coords(pdb_id: str, chain_id: str, device="cuda"):
    # Load protein chain
    print(f"Loading protein chain for PDB ID: {pdb_id}, Chain ID: {chain_id}")
    chain = ProteinChain.from_rcsb(pdb_id, chain_id)
    
    # Backbone atom indices in atom37 convention
    backbone_indices = [0, 1, 2]  # N, CA, C
    
    # atom37_mask: shape (L, 37), True if atom is present
    backbone_mask = chain.atom37_mask[:, backbone_indices]  # shape (L, 3)
    
    # A residue is present if all backbone atoms are present
    present_res_mask = backbone_mask.all(axis=1)  # shape (L,)
    print(f"Number of residues with all backbone atoms present: {present_res_mask.sum()} out of {len(present_res_mask)}")
    print(type(present_res_mask))

    # Get backbone coordinates for present residues only
    coords = chain.atom37_positions[present_res_mask][:, backbone_indices, :]  # shape (num_present, 3, 3)
    coords = torch.from_numpy(coords).float().unsqueeze(0).to(device)  # shape (1, num_present, 3, 3)
    print(f"Filtered backbone coordinate shape: {coords.shape}")
    
    return coords, present_res_mask


def prepare_structure_track_input(
    pdb_id: str, chain_id: str, device="cuda"
) -> dict[str, Any]:
    """Prepares inputs to the model with default values for all tracks except sequence"""

    structure_coords, _ = get_backbone_coords(pdb_id, chain_id, device)
    L = structure_coords.size(1)

    defaults = lambda x, tok: (
        torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
    )

    sequence_tokens = defaults(None, C.SEQUENCE_PAD_TOKEN)

    ss8_tokens = defaults(None, C.SS8_PAD_TOKEN)
    sasa_tokens = defaults(None, C.SASA_PAD_TOKEN)
    average_plddt = defaults(None, 1).float()
    per_res_plddt = defaults(None, 0).float()
    chain_id_tensor = defaults(None, 0)
    residue_annotation_tokens = torch.full(
        (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
    )
    function_tokens = torch.full(
        (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
    )
    affine, affine_mask = build_affine3d_from_coordinates(structure_coords)
    structure_tokens = defaults(None, C.STRUCTURE_MASK_TOKEN)
    structure_tokens = (
        structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
        .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
        .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
        .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
        .masked_fill(
            sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
            C.STRUCTURE_CHAINBREAK_TOKEN,
        )
    )

    return {
        "sequence_tokens": sequence_tokens,
        "structure_tokens": structure_tokens,
        "ss8_tokens": ss8_tokens,
        "sasa_tokens": sasa_tokens,
        "average_plddt": average_plddt,
        "per_res_plddt": per_res_plddt,
        "chain_id": chain_id_tensor,
        "residue_annotation_tokens": residue_annotation_tokens,
        "function_tokens": function_tokens,
        "structure_coords": structure_coords,
        "affine": affine,
        "affine_mask": affine_mask,
    }

@torch.no_grad()
def get_layer_embedding(
   pdb_id: str, chain_id: str, model: ESM3, layer: int, device="cuda"
) -> dict[int, torch.Tensor]:
    '''Get structure token embeddings from a specific layer'''

    inputs = prepare_structure_track_input(pdb_id, chain_id, device)
    affine = inputs.pop("affine")
    affine_mask = inputs.pop("affine_mask")
    
    x = model.encoder(
        inputs["sequence_tokens"], inputs["structure_tokens"],
        inputs["average_plddt"], inputs["per_res_plddt"],
        inputs["ss8_tokens"], inputs["sasa_tokens"],
        inputs["function_tokens"], inputs["residue_annotation_tokens"]
    )
    
    sequence_id = None
    chain_id_tensor = torch.ones(size=(1, x.size(1)), dtype=torch.long, device=x.device)
    
    # Run through ALL blocks sequentially
    for i, block in enumerate(model.transformer.blocks):
        x = block(x, sequence_id, affine, affine_mask, chain_id_tensor)
        if i == layer:
            break
    
    # Get structure tokens
    structure_logits = model.output_heads.structure_head(x)
    structure_tokens = structure_logits.argmax(dim=-1) 
    
    return structure_tokens