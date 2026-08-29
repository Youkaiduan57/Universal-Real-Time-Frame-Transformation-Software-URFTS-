import numpy as np
from frame_processor import FrameProcessor

class Backend:
    def process(self,frame): return frame.copy()

def test_shader_processing_reports_wall_time():
    processor=FrameProcessor(processing_backend=Backend())
    result=processor.process(np.zeros((4,4,3),dtype=np.uint8))
    assert result.shape==(4,4,3)
    assert processor.last_postprocessing_ms >= 0


def test_refinement_bypass_flat_fields_and_detail():
    from frame_processor import refine_output
    import pytest
    image=np.full((8,8,3),100,dtype=np.uint8)
    assert refine_output(image,0) is image
    np.testing.assert_array_equal(refine_output(image,.12),image)
    image[4,4]=140
    before=image.copy()
    result=refine_output(image,.12)
    assert result[4,4,0]>image[4,4,0]
    assert result.dtype==np.uint8
    assert result.shape==image.shape
    np.testing.assert_array_equal(refine_output(image,.12),result)
    np.testing.assert_array_equal(image,before)
    with pytest.raises(ValueError): refine_output(image,.3)
