# Numerical audit: CFFNN–MADM

**Trạng thái: chưa reproduce được paper. Table 3 FAIL; output, scalar score,
ranking và sensitivity bị chặn.** Không có PyTorch/TensorFlow, quá trình train,
dataset tự tạo, hay final score được chèn vào thuật toán.

Nguồn: Aliya Fahmi, Saeed Islam, Muhammad Arshad Shehzad Hassan, Ishtiaq Ali,
“Multiple attribute decision-making based on fuzzy neural network under complex
Fermatean fuzzy data”, *Signal, Image and Video Processing* (2025), **19:1171**,
DOI **10.1007/s11760-025-04660-0**.
Đã đọc [PDF trong workspace](../docs/Multiple-attribute-decision-making-based-on-fuzzy-neural-network-under-complex-Fermatean.pdf),
đối chiếu trực tiếp hình trang 3–10 và 12 trước khi implement.
Số trang dưới đây là số trang PDF (1–13).

Hướng dẫn [docs/cffnn_reproduction.codex](../docs/cffnn_reproduction.codex),
Phase 1.5, yêu cầu: **“Nếu Table 3 chưa reproduce được: DỪNG.
Không tiếp tục final ranking bằng workaround.”** Vì vậy, không lấy Table 3/4
in sẵn thay cho hidden/output do code tính, không thử điều chỉnh score để khớp Z.

## Chạy

Từ thư mục `cffnn-reproduction`, với Python 3.13 đã được kiểm tra:

Runtime/test dependencies được pin theo môi trường đã chạy: NumPy 2.5.3,
pytest 9.1.1. PyMuPDF 1.28.2 chỉ dùng để đọc/render PDF khi audit, không cần
cho việc chạy source hoặc tests.

```powershell
..\.venv\Scripts\python reproduce.py
..\.venv\Scripts\python -m pytest -q -m "not paper"
..\.venv\Scripts\python -m pytest -q --tb=no
```

Nếu tạo môi trường mới ở thư mục gốc workspace:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r cffnn-reproduction/requirements.txt
```

`reproduce.py` luôn ghi [báo cáo 64 thành phần](reports/reproduction.md) và
[JSON đầy đủ](reports/reproduction.json). Exit code **1** hiện tại là kết quả
đúng của kiểm tra reproduction bị FAIL, không phải crash bị giấu.

Hai loại kiểm thử được tách bằng marker `paper`:

- Kiểm thử implementation: **74 PASS**. Chứng minh code tuân theo từng công thức/
  giả thuyết được ghi rõ, không chứng minh paper đã được reproduce.
- Toàn bộ suite: **74 PASS, 64 FAIL, 15 SKIP**. 64 FAIL kiểm tra riêng từng thành
  phần Table 3. 15 SKIP là 4 ô Table 4, 4 score Z, ranking và 6 sensitivity;
  đều bị chặn bởi Table 3. Không sử dụng `xfail` để làm suite xanh.

Mặc định so sánh số với `abs=1e-4, rel=0`; validation toán học dùng `1e-12`
để không chấp nhận dữ liệu sai miền. Một test chống overflow dùng relative
tolerance vì giá trị cần kiểm tra nhỏ hơn `1e-4` rất nhiều.

## Mapping công thức → code

| Nguồn | File / hàm | Cách đọc |
|---|---|---|
| Def.2.3 đầu tiên, p.3 | `src/cff.py`, `CFFNumber` | 4 số thực riêng biệt; phase chuẩn hóa theo vòng, không phải radian |
| §2.1 p.3 | `src/schweizer_sklar.py` | SS t-norm, dual t-conorm, limit Γ=0 |
| Def.3.1 p.4 | `src/operations.py` | add, multiply, scalar multiply, power; ba interpretation được chọn tường minh |
| Def.4.1.1 p.4 | `aggregation.cffsstwa` | fold `⊕_j (w_j d_j)` |
| Def.4.2.1 p.5 | `aggregation.cffsstowa` | bản in lặp định nghĩa WA; ordering chỉ có khi caller cung cấp key |
| Def.4.3.1 pp.5–6 | `aggregation.cffssthwa` | bản in cũng là fold WA; hybrid thông thường là giả thuyết opt-in |
| Thm.4.3.2 p.6 | `aggregation.printed_hwa_product` | giữ riêng biểu thức tích, ρ và ϑ tường minh, kiểm tra cả chỉ số 1 và j |
| Tables 1–2 p.8 | `data/paper_case.py` | 2×4×4×4 giá trị, HIC trước LIS |
| Hidden layer pp.6–9 | `madm.hidden_layer` | giả thuyết gộp hai bảng theo từng ô để có bảng 4×4 |
| Table 3 pp.8–9 | `madm.require_table3` | gate toàn bộ 64 thành phần |
| Output p.9 | `madm.output_layer` | WA theo trọng số (0.23,0.24,0.25,0.28), không thực thi khi gate FAIL |
| Def.2.3 thứ hai p.3 | `src/score.py` | score **complex** đúng dạng in; không tự chọn phép chiếu ra số thực |
| §5.7 pp.11–12 | `madm.sensitivity_analysis` | guard; chưa có baseline PASS và chưa biết ý nghĩa q |

Expected Table 3/4, scores và sensitivity chỉ nằm trong dữ liệu đối chiếu/test/
report. Không có expected nào tham gia tính hidden layer. Không làm tròn trước
khi in. Không đồng nhất phase với magnitude trong datatype hoặc thuật toán.

### Schweizer–Sklar và ba interpretation của operational laws

Ký hiệu `g=Γ`:

```text
T_g(a,b) = (a^g + b^g - 1)^(1/g), g<0
S_g(a,b) = 1 - T_g(1-a,1-b)
T_0(a,b) = a*b; S_0(a,b) = a+b-a*b

Q_g(a,b) = cbrt(S_g(a^3,b^3))
R_g,w(x) = (w*x^g + (w-1))^(1/g)        [in nguyên văn]
R*_g,w(x) = (1 + w*(x^g-1))^(1/g)       [giả thuyết sửa dấu]
```

Áp dụng riêng cho magnitude và phase:

| Law | `literal` (Def.3.1) |
|---|---|
| add | mu=`Q_g(mu1,mu2)`; nu=`T_g(nu1,nu2)` |
| multiply | mu=`T_g(mu1,mu2)`; nu=`Q_g(nu1,nu2)` |
| scalar w*x | mu=`cbrt(1-R_g,w(1-mu^3))`; nu=`R_g,w(nu)` |
| power x^w | mu=`R_g,w(mu)`; nu=`cbrt(1-R_g,w(1-nu^3))` |

- `literal` là mặc định API: giữ dấu `+(w−1)` và nhánh T không có cube như bản in.
  Nếu kết quả ra ngoài miền thì raise `ValueError`; không dùng trị tuyệt đối,
  không bỏ phần ảo, không cắt giá trị sai thành CFF hợp lệ.
- `sign_corrected`: **chỉ** thay R bằng R*. Đây là giả thuyết để so sánh,
  không được gọi là công thức paper chính xác. Dấu này phù hợp với việc lặp
  phép cộng: `2*x` phải dùng `2*x^g−1`, còn bản in scalar dùng `2*x^g+1`.
- `fermatean`: dùng R* và thêm cubes/cube roots cho nhánh T tương ứng.
  Đây là một giả thuyết khác, không lẫn với cách đọc nguyên văn.

`positive='reject'` giữ miền §2.1. Với g>0, `positive='literal'` thử biểu thức
không cắt ở các input còn hợp lệ; `positive='truncate'` là extension được
ghi rõ `max(0, a^g+b^g−1)^(1/g)`. Không đổi dấu tham số âm thầm.

Ở `sign_corrected`, fold WA tương đương:

```text
mu = cbrt(1 - (sum_j w_j*(1-mu_j^3)^g)^(1/g))
nu = (sum_j w_j*nu_j^g)^(1/g)
```

Ở `fermatean`, thay nhánh nu bằng
`cbrt((sum_j w_j*(nu_j^3)^g)^(1/g))`.
Tests dùng biểu thức closed form độc lập này để đối chiếu fold.

### OWA/HWA chưa được paper phân biệt đầy đủ

Definitions 4.1.1, 4.2.1, 4.3.1 đều in cùng fold, không có permutation, vector
trọng số thứ hai hay quy tắc xếp thứ tự. API mặc định giữ đúng sự trùng lặp đó.
Chỉ khi cung cấp `key`, OWA mới sort giảm dần như một giả thuyết. Chỉ khi cung
cấp cả `order_weights` và `key`, HWA mới tính giả thuyết hybrid thông thường:
`OWA(sort(n*w_i*x_i), order_weights)`. Không giả vờ đó là chi tiết có trong paper.

Theorem 4.3.2 lại in một công thức khác:

```text
mu = cbrt(1 - product_j [rho*(1-L_1^3)^theta + (rho-1)]^(1/theta))
nu = [rho*product_j M_1^theta + (rho-1)]^(1/theta)
```

Code `printed_hwa_product` giữ biểu thức này với chỉ số lặp `1`, hoặc thử thay
`1→j` bằng option `indexing='indexed'`. ρ, θ phải được cung cấp rõ. Ở ρ=1,
input lặp x cho mu=`cbrt(1-(1-mu^3)^n)`, nu=`nu^n`, **không idempotent** như
Theorem 4.3.3 tuyên bố. Đây là counterexample, không phải một phép sửa paper.

## Kết quả thực tế

Paper không nêu baseline Γ/ρ/ϑ và expert weights đủ để chạy duy nhất một
baseline. Báo cáo chọn **một cấu hình chẩn đoán cố định**, không fit vào target:

```text
Gamma = -0.1
expert_weights = (0.5, 0.5)
interpretation = sign_corrected
topology = gộp Table 1 và Table 2 cùng HIC/LIS
```

| Table 3 HIC1/LIS1 | Expected | Actual chẩn đoán | Sai số tuyệt đối |
|---|---:|---:|---:|
| mu magnitude | 0.1091 | 0.079588269 | 0.029511731 |
| mu phase | 0.0245 | 0.079588269 | 0.055088269 |
| nu magnitude | 0.3881 | 0.088797819 | 0.299302181 |
| nu phase | 0.0212 | 0.088797819 | 0.067597819 |

**64/64 thành phần Table 3 FAIL**, max absolute error **0.942200374**.
Trường hợp `literal`, cùng Γ và weights, gặp lỗi miền ở scalar trước khi
có hidden layer hợp lệ. Báo cáo không đánh tráo nó với cấu hình chẩn đoán.

| Output | Expected | Actual | Sai số | Trạng thái |
|---|---:|---|---|---|
| Z1 | 0.0199 | chưa tính | N/A | BLOCKED |
| Z2 | 0.5478 | chưa tính | N/A | BLOCKED |
| Z3 | 0.2979 | chưa tính | N/A | BLOCKED |
| Z4 | 0.5148 | chưa tính | N/A | BLOCKED |

**Chưa reproduce được `Z2 > Z4 > Z3 > Z1`.** Không có actual ranking hoặc
sensitivity scores để báo cáo. Các expected sensitivity q=0.1,…,0.6 được
chép đầy đủ từ p.12 vào fixture, không dùng làm kết quả tính toán.

## Bằng chứng mismatch không chỉ do thiếu tuning

1. **Đối xứng magnitude–phase.** Tất cả 32 input có mu magnitude bằng mu phase.
   Các luật in trong paper dùng cùng biểu thức cho hai nhánh; các interpretation
   đã thử cũng vậy. Bởi thế mọi chuỗi operation/aggregation dùng cùng weights
   và parameter cho hai nhánh đều giữ mu magnitude=mu phase, kể cả đổi topology.
   Nhưng **14/16 ô Table 3** yêu cầu hai giá trị khác nhau hơn `2e-4`.
   Muốn phá đối xứng phải thêm phép biến đổi/weights riêng cho phase mà paper
   chưa cung cấp. Không được tự thêm.
2. **HIC1/LIS1** có hai input `(0.1,0.1,0.3,0.3)` và
   `(0.02,0.02,0.03,0.03)`. Cả mu lẫn nu đều đối xứng. Với nu output chung x,
   `max(|x−0.3881|, |x−0.0212|) >= (0.3881−0.0212)/2 = 0.18345`.
   Bất đẳng thức này độc lập việc chọn parameter và expert weights cho cellwise
   fusion. Weighted mean đơn điệu/idempotent còn phải nằm giữa hai input;
   target mu magnitude 0.1091 > 0.1 và nu magnitude 0.3881 > 0.3 vi phạm bound.
3. **Dữ liệu Table 3 không phải CFF hợp lệ** theo điều kiện componentwise
   Fermatean được nêu rõ trong datatype:

   | Ô | Nhánh | mu^3+nu^3 |
   |---|---|---:|
   | HIC3/LIS2 | magnitude | 1.282028385 |
   | HIC4/LIS1 | magnitude | 1.000884140 |
   | HIC4/LIS2 | magnitude và phase | 1.329847705 |
   | HIC4/LIS4 | magnitude | 1.098313728 |
   | HIC4/LIS4 | phase | 1.642340226 |

   Hạ cả hai thành phần mỗi ô xuống `1e-4` vẫn cho tổng >1. Vì vậy không có CFF
   hợp lệ nào khớp những ô đó trong tolerance yêu cầu. Targets được lưu dạng
   tuple raw để không lách validation của datatype.
4. **Idempotency:** Table 1 và 2 giống nhau ở HIC2/LIS4, HIC3/LIS3,
   HIC4/LIS3, nhưng Table 3 không bằng input ở các ô này. Điều này trái với
   Theorem 4.3.3 nếu hidden layer là aggregation của hai expert cùng ô.

## Các interpretation đã thử

`src/diagnostics.py` thử trước khi đi tiếp output, không tối ưu/fitting theo Z:

- 198 cấu hình fold: ba interpretation; Γ trong
  `(-1,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,1)`;
  expert weights `(0.25,0.75)`, `(0.5,0.5)`, `(0.75,0.25)`;
  với Γ>0 thử cả bản không truncate và extension truncate.
- 30 cấu hình công thức tích Theorem 4.3.2: ρ∈{0.5,1,2},
  θ∈{−0.6,−0.1,0.1,0.6,1}, cả chỉ số 1 và j.
- 36 cấu hình hybrid thông thường: sign-corrected/fermatean; Γ=−0.1;
  ba expert weights, ba positional weights; hai key chẩn đoán
  `mu_mag^3−nu_mag^3` và trung bình hai hiệu cube magnitude/phase.
  Hai key này **không** được tuyên bố là score Definition 2.3.

Tổng **264**: **203 MISMATCH**, **61 DOMAIN_ERROR**, **0 PASS**.
Sai số max nhỏ nhất trong tập thử vẫn là **0.833851089**; đó chỉ là thống kê
trong tập thử, không phải nghiệm tối ưu hoặc bằng chứng đã duyệt mọi parameter.
Các bằng chứng đối xứng/miền CFF ở trên mới là lý do không thể đạt toàn Table 3
trong phạm vi các luật đã công bố. Việc dùng ±0.1,…,±0.6 ở đây chỉ là diagnostic
hidden layer, **không phải** thực hiện sensitivity analysis cuối pipeline.

## Notation/thông tin chưa nhất quán

- §2.1 p.3 yêu cầu Γ<0, Def.3.1 p.4 lại Γ>0, §5.5 p.11 nói p→+∞ cho min
  nhưng không ánh xạ convention; sensitivity đổi tên thành q.
- Scalar/power dùng `+(n−1)`, mâu thuẫn phép lặp binary dùng `−1`.
- Nhánh T không cube/cube root, nhánh S có cube/cube root. Giữ nguyên ở literal;
  chỉ thử thêm cubes dưới tên hypothesis riêng.
- Definitions 4.1/4.2/4.3 trùng nhau; theorems dùng ρ không định nghĩa, ϑ vừa là
  vector weight vừa là số mũ scalar; product lặp L1/M1; công thức theorem và
  definition khác nhau, proof idempotency không đúng cho biểu thức tích.
- Paper chưa chỉ rõ expert weights, baseline parameter, cách dùng hai bảng
  input, hoặc cách pseudocode H_i gồm một số/alternative tạo ra Table 3 4×4.
  Không tự coi criterion weights 4 phần tử là expert weights 2 phần tử.
- Table 1 HIC1/LIS2 in nu magnitude=0.1, nu phase=0.7: đã **giữ nguyên**.
  Không tự “sửa typo” thành 0.7 ở cả hai nhánh.
- Def.2.3 về CFF ghi so sánh complex với `[0,1]`, không có order chuẩn trên
  complex. Datatype dùng ràng buộc thực riêng cho magnitude và phase, và ghi rõ
  đó là interpretation. Cùng số Definition 2.3 được dùng lại cho score.
- Score in ở p.3 là `D^3 exp(i2πD^3) − E^3 exp(i2πE^3)`. Sau khi tách phase
  độc lập, nó vẫn là **số phức**; paper không nêu cách chuyển thành scalar Z
  hoặc cách xếp thứ tự số phức. `score()` trả complex; `scalar_scores()` chặn.
- Table 4 p.10 gắn nhãn LIS1–LIS4 trong khi score/ranking nói về alternatives.
  Phần text output §5/pseudocode dùng HWA, case study p.9 dùng WA. Chưa có
  bảng output do code tính để xác nhận cách ánh xạ; test Table 4 chỉ giữ thứ tự
  hàng in, chưa tuyên bố LIS tương ứng HIC.

Để tiếp tục một reproduction đầy đủ cần dữ liệu Table 3 hợp lệ/đính chính,
công thức operational/aggregation nhất quán, quy tắc hidden topology và weights,
baseline Γ/ρ/ϑ, score thực có thể sắp thứ tự, mapping Table 4 và ý nghĩa q.
Không triển khai phase trainable neural network trong project này.
