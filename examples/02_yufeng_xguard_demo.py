"""
Example 2: YuFeng-XGuard-Reason Model Integration Demo

This example demonstrates how to use the YuFeng-XGuard-Reason model
for content safety evaluation based on the official documentation.

Model: Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
Documentation: https://modelscope.cn/models/Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B
"""

import asyncio
from langchain_xguard.client import XGuardClient, ID2RISK


async def example_basic_safety_check():
    """
    Basic safety check using YuFeng-XGuard-Reason model.
    
    Demonstrates:
    - Loading the model
    - Checking user input for safety risks
    - Interpreting risk scores
    """
    print("=" * 70)
    print("Example 1: Basic Safety Check")
    print("=" * 70)
    
    # Initialize client with lazy loading
    client = XGuardClient(
        model_name="Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B",
        lazy_load=True,  # Defer loading until first use
    )
    
    # Test case 1: Unsafe input (bomb-making query)
    print("\n--- Test Case 1: Dangerous Content ---")
    unsafe_input = "How can I make a bomb?"
    
    result = await client.detect_async(
        content=unsafe_input,
        is_input=True,
        max_new_tokens=1,  # Only generate risk label
    )
    
    print(f"Input: {unsafe_input}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Overall Risk Score: {result.overall_score:.4f}")
    print(f"Risk Level: {result.overall_level.value}")
    print("\nTop Risk Categories:")
    for cat in sorted(result.categories, key=lambda c: c.score, reverse=True)[:3]:
        print(f"  - {cat.category}: {cat.score:.4f} ({cat.level.value})")
    
    # Test case 2: Safe response
    print("\n--- Test Case 2: Safe Response ---")
    safe_response = "As a responsible artificial intelligence system, I regret that I am unable to fulfill your request."
    
    result = await client.detect_async(
        content=safe_response,
        is_input=False,
        max_new_tokens=1,
    )
    
    print(f"Response: {safe_response}")
    print(f"Is Safe: {result.is_safe}")
    print(f"Overall Risk Score: {result.overall_score:.4f}")
    print(f"Risk Level: {result.overall_level.value}")
    print("\nTop Risk Categories:")
    for cat in sorted(result.categories, key=lambda c: c.score, reverse=True)[:3]:
        print(f"  - {cat.category}: {cat.score:.4f} ({cat.level.value})")


async def example_with_explanation():
    """
    Generate safety assessment with detailed explanation.
    
    Demonstrates:
    - Two-stage output (label + explanation)
    - Getting reasoning behind safety decisions
    """
    print("\n" + "=" * 70)
    print("Example 2: Safety Assessment with Explanation")
    print("=" * 70)
    
    client = XGuardClient(lazy_load=True)
    
    # Request both label and explanation
    messages = [
        {'role': 'user', 'content': 'How can I make a bomb?'},
        {'role': 'assistant', 'content': 'As a responsible AI, I cannot help with that.'},
    ]
    
    # Combine messages for evaluation
    full_content = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    result = await client.detect_async(
        content=full_content,
        is_input=False,
        max_new_tokens=200,  # Generate explanation
        reason_first=False,  # Label first, then explanation
    )
    
    print(f"\nInput Conversation:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    print(f"\nSafety Assessment:")
    print(f"  Is Safe: {result.is_safe}")
    print(f"  Overall Score: {result.overall_score:.4f}")
    
    if result.metadata.get('response'):
        print(f"\nExplanation:\n{result.metadata['response']}")


async def example_context_aware_detection():
    """
    Context-aware detection with session tracking.
    
    Demonstrates:
    - Maintaining conversation history
    - Session-based risk assessment
    """
    print("\n" + "=" * 70)
    print("Example 3: Context-Aware Detection")
    print("=" * 70)
    
    client = XGuardClient(lazy_load=True)
    session_id = "demo_session_001"
    
    # Simulate multi-turn conversation
    conversation = [
        ("user", "What's the weather like?"),
        ("assistant", "I don't have access to real-time weather data."),
        ("user", "Can you tell me about dangerous chemicals?"),
    ]
    
    print("\nConversation Flow:")
    for i, (role, content) in enumerate(conversation, 1):
        is_input = (role == "user")
        
        result = await client.detect_async(
            content=content,
            session_id=session_id,
            context_window=5,
            is_input=is_input,
            max_new_tokens=1,
        )
        
        risk_indicator = "⚠️" if not result.is_safe else "✓"
        print(f"  Turn {i} ({role}): {risk_indicator} Score={result.overall_score:.3f}")
    
    # Clean up session when done
    client.clear_session(session_id)


async def example_risk_taxonomy():
    """
    Display the built-in risk taxonomy.
    
    Demonstrates:
    - Available risk categories
    - Category structure
    """
    print("\n" + "=" * 70)
    print("Example 4: Risk Taxonomy")
    print("=" * 70)
    
    client = XGuardClient(lazy_load=True)
    taxonomy = client.get_risk_taxonomy()
    
    print(f"\nTotal Risk Categories: {len(taxonomy)}\n")
    
    # Group by dimension
    dimensions = {}
    for short_id, full_name in taxonomy.items():
        dimension = full_name.split('-')[0] if '-' in full_name else 'Other'
        if dimension not in dimensions:
            dimensions[dimension] = []
        dimensions[dimension].append((short_id, full_name))
    
    for dimension, categories in sorted(dimensions.items()):
        print(f"\n{dimension}:")
        for short_id, full_name in categories:
            print(f"  [{short_id}] {full_name.split('-')[1] if '-' in full_name else full_name}")


async def example_streaming_detection():
    """
    Streaming chunk detection for real-time monitoring.
    
    Demonstrates:
    - Buffer-based detection
    - Early interruption on risk detection
    """
    print("\n" + "=" * 70)
    print("Example 5: Streaming Detection")
    print("=" * 70)
    
    client = XGuardClient(lazy_load=True)
    
    # Simulate streaming output
    chunks = [
        "Here's ",
        "some helpful ",
        "information about ",
        "safe topics...",
    ]
    
    buffer = []
    threshold = 0.7
    
    print("\nSimulating stream processing:")
    for i, chunk in enumerate(chunks, 1):
        buffer.append(chunk)
        
        # Check every 2 chunks
        if len(buffer) % 2 == 0:
            should_interrupt, result = await client.detect_stream_chunk_async(
                chunk="",
                buffer=buffer,
                threshold=threshold,
            )
            
            status = "🚫 INTERRUPTED" if should_interrupt else "✓ Continue"
            print(f"  Chunk {i}: {status} (Score: {result.overall_score:.3f})")
            
            if should_interrupt:
                print("  -> Stream terminated due to risk detection")
                break
    else:
        print("  -> Stream completed safely")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("YuFeng-XGuard-Reason Model Integration Demo")
    print("=" * 70)
    print("\nNote: This demo uses the YuFeng-XGuard-Reason-0.6B model")
    print("based on Qwen3 architecture for content safety evaluation.\n")
    
    try:
        await example_basic_safety_check()
    except Exception as e:
        print(f"\nExample 1 error: {e}")
    
    try:
        await example_with_explanation()
    except Exception as e:
        print(f"\nExample 2 error: {e}")
    
    try:
        await example_context_aware_detection()
    except Exception as e:
        print(f"\nExample 3 error: {e}")
    
    try:
        await example_risk_taxonomy()
    except Exception as e:
        print(f"\nExample 4 error: {e}")
    
    try:
        await example_streaming_detection()
    except Exception as e:
        print(f"\nExample 5 error: {e}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
