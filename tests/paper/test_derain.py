def run_derain():
    pass


def test_Rain100H():
    psnr, ssim = run_derain()
    assert abs(psnr - 31.08) < 0.01
    assert abs(ssim - 0.897) < 0.001


def test_Rain100H_with_init():
    psnr, ssim = run_derain()
    assert abs(psnr - 31.62) < 0.01
    assert abs(ssim - 0.905) < 0.001


def test_Test1200():
    psnr, ssim = run_derain()
    assert abs(psnr - 32.95) < 0.01
    assert abs(ssim - 0.913) < 0.001


def test_Test1200_with_init():
    psnr, ssim = run_derain()
    assert abs(psnr - 33.25) < 0.01
    assert abs(ssim - 0.926) < 0.001
