"""Submission packaging checks; no tracking, labels or model payloads loaded."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='d74b990829aa0ec69a141400129d92df0f4d17573749b15696fd6a026adf7a44'


class MetadataOnly(ast.NodeTransformer):
    def visit_Module(self,node):
        self.generic_visit(node)
        if isinstance(node.body[0],ast.Expr)and isinstance(node.body[0].value,ast.Constant):node.body[0].value.value='DOCUMENTATION'
        return node
    def visit_Constant(self,node):
        if node.value in ('local_unpromoted_research_candidate','selected_for_official_submission'):node.value='SELECTION_ROLE'
        if node.value in ('Unexpected candidate role or source-NN dependency','Unexpected submission role or source-NN dependency'):node.value='ROLE_ERROR_MESSAGE'
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        node.keywords=[k for k in node.keywords if k.arg not in {'promoted','submitted','published','selected_for_official_submission','external_submission_state_not_measured'}]
        return node


class SubmissionMetadata(unittest.TestCase):
    def test_all_152_original_scientific_assets_are_exact(self):
        original=json.loads((ROOT/'inference.json').read_text())['files']
        self.assertEqual(len(original),152)
        for name,digest in original.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_entry_changes_only_selection_documentation_and_receipt_metadata(self):
        old=ROOT/'provenance/strong1-driver-original.py.txt'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),'39b18ec02af4d90e2b414eff833c189a99b024f120ab3754e99f03078b04e178')
        left=MetadataOnly().visit(ast.parse(old.read_text()));right=MetadataOnly().visit(ast.parse((ROOT/'reproduce_strong_scale1.py').read_text()))
        self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False))
        self.assertEqual(hashlib.sha256((ROOT/'second_zero/strong_scale1_candidate.py').read_bytes()).hexdigest(),'85c5797645f4deb5418ce41cf7e4bea68eb4f3d748df7adbd6391c349d0ff078')

    def test_selected_contract_and_all_entry_hashes(self):
        config=json.loads((ROOT/'strong1.json').read_text())
        self.assertEqual(config['status'],'selected_for_official_submission');self.assertFalse(config['source_NN_used'])
        self.assertEqual(config['output_sha256'],EXPECTED)
        self.assertEqual(config['policy'],{'alpha':.5,'apply_source':False,'id':'strong_scale1','parent_confidence':.98,'status_mode':'joint','transition_scale':1.})
        for name,digest in {**config['preserved_base_files'],**config['additional_files']}.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_no_tracking_reference_prediction_or_secret_file_assets(self):
        forbidden={'.csv','.npz','.npy','.xml','.mp4','.mov','.zip','.pem','.key','.p12','.pfx','.sqlite','.db'}
        for path in ROOT.rglob('*'):
            if not path.is_file():continue
            self.assertNotIn(path.suffix.lower(),forbidden,str(path))
            self.assertFalse(path.name in {'.env','id_rsa','id_ed25519'}or path.name.endswith('_atd.txt'),str(path))


if __name__=='__main__':unittest.main()
