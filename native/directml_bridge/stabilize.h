// Inference-resolution confidence compositor. No frame-pixel CPU readback.
// HUD edges use a conservative gradient mask, not CPU Canny parity.
static const char* kStabilizeShader = R"(
StructuredBuffer<float> Generated : register(t0);
StructuredBuffer<float> First : register(t1);
StructuredBuffer<float> Second : register(t2);
Texture2D<float4> Adjustment : register(t3);
RWTexture2D<float4> Result : register(u0);
cbuffer Params : register(b0) { uint Width; uint Height; uint OutWidth; uint OutHeight;
    uint LogicalWidth; uint LogicalHeight; uint Reserved0; uint Reserved1; };
int2 bound(int2 p) { return clamp(p, int2(0,0), int2(LogicalWidth-1,LogicalHeight-1)); }
float3 first(int2 p) { p=bound(p); uint n=p.y*Width+p.x, s=Width*Height;
    return float3(First[n],First[s+n],First[2*s+n]); }
float3 second(int2 p) { p=bound(p); uint n=p.y*Width+p.x, s=Width*Height;
    return float3(Second[n],Second[s+n],Second[2*s+n]); }
float3 generated(int2 p) { p=bound(p); uint n=p.y*Width+p.x, s=Width*Height;
    float3 g=float3(Generated[n],Generated[s+n],Generated[2*s+n]);
    return all(isfinite(g)) ? saturate(g) : (first(p)+second(p))*0.5; }
float maxrgb(float3 x) { return max(x.x,max(x.y,x.z)); }
float motion(int2 p) { return maxrgb(abs(first(p)-second(p))); }
bool stable(int2 p) {
    bool good=true;
    [unroll] for(int y=-1;y<=1;y++) [unroll] for(int x=-1;x<=1;x++)
        good = good && motion(p+int2(x,y)) <= 12.0/255.0;
    return good;
}
groupshared float4 sums[256];
[numthreads(256,1,1)]
void Correct(uint lane: SV_GroupIndex) {
    float4 total=0;
    for(uint n=lane;n<LogicalWidth*LogicalHeight;n+=256) {
        int2 p=int2(n%LogicalWidth,n/LogicalWidth);
        if(stable(p)) total += float4((first(p)+second(p))*0.5-generated(p),1);
    }
    sums[lane]=total; GroupMemoryBarrierWithGroupSync();
    for(uint stride=128;stride>0;stride/=2) {
        if(lane<stride) sums[lane]+=sums[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(lane==0) {
        float3 delta=sums[0].w>=16 ? clamp(sums[0].xyz/max(sums[0].w,1),-4.0/255.0,4.0/255.0) : 0;
        if(maxrgb(abs(delta))<0.5/255.0) delta=0;
        Result[uint2(0,0)]=float4(delta,0);
    }
}
[numthreads(8,8,1)]
void Composite(uint3 id: SV_DispatchThreadID) {
    if(id.x>=LogicalWidth || id.y>=LogicalHeight) return;
    int2 p=int2(id.xy);
    float3 delta=Adjustment.Load(int3(0,0,0)).rgb;
    bool fallback=stable(p);
    [unroll] for(int y=-1;y<=1;y++) [unroll] for(int x=-1;x<=1;x++) {
        int2 q=p+int2(x,y);
        float3 midpoint=(first(q)+second(q))*0.5;
        float3 g=saturate(generated(q)+delta);
        fallback = fallback || maxrgb(abs(g-midpoint)) > motion(q)*0.75+8.0/255.0;
        // Protect persistent HUD/text edges, dilated one inference pixel.
        float edge=max(maxrgb(abs(first(q+int2(1,0))-first(q-int2(1,0)))),
                       maxrgb(abs(first(q+int2(0,1))-first(q-int2(0,1)))));
        fallback = fallback || (motion(q)<=18.0/255.0 && edge>48.0/255.0);
    }
    Result[id.xy]=float4(fallback ? (first(p)+second(p))*0.5 : saturate(generated(p)+delta), fallback ? 1 : 0);
}
)";
